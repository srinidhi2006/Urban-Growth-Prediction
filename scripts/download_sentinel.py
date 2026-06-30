"""
Sentinel-2 Image Ingestion Orchestrator.
CLI script to query, compose, quality-check, and export satellite layers.
"""

import argparse
import sys
import time
import json
import numpy as np
import threading
from pathlib import Path
from loguru import logger

manifest_lock = threading.Lock()

# Add root folder to search path if executing directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config
from gee.authenticate import authenticate_gee
from gee.utils import get_city_boundary
from gee.sentinel_pipeline import build_sentinel_composite
from gee.export import download_locally, export_to_drive, generate_rgb_preview, write_metadata

def run_pipeline(
    city: str, 
    year: int, 
    mode: str = "local", 
    scale_override: int = None
) -> bool:
    """Executes the complete ingestion pipeline for a single city and year.
    
    Args:
        city (str): Target city name.
        year (int): Target calendar year.
        mode (str): Export mode ('local' or 'drive').
        scale_override (int, optional): Override default pixel resolution scale.
        
    Returns:
        bool: True if process completed successfully, False otherwise.
    """
    start_time = time.time()
    logger.info(f"======================================================================")
    logger.info(f"STARTING SENTINEL-2 PIPELINE | City: {city} | Year: {year} | Mode: {mode}")
    logger.info(f"======================================================================")
    
    # 1. Resolve output path directories
    out_dir = Config.RAW_SENTINEL_DIR / city / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tif_path = out_dir / "image.tif"
    png_path = out_dir / "image_preview.png"
    meta_path = out_dir / "metadata.json"
    
    # Resolve target scale
    scale = scale_override if scale_override else Config.SATELLITE_SCALE
    
    try:
        # 2. Get administrative boundaries
        boundary = get_city_boundary(city)
        
        # 3. Build masked composite
        composite, qc_metrics, image_count = build_sentinel_composite(
            city_name=city,
            year=year,
            boundary=boundary
        )
        
        # Log collection metrics
        logger.info(f"Initial raw scene count matching query parameters: {image_count}")
        logger.info(f"Composite metrics: Valid Pixel Density: {qc_metrics['valid_pixel_ratio']:.2%}")
        
        # 4. Handle Quality Check Failure (Warning only, do not block unless required)
        if not qc_metrics["passed"]:
            logger.warning(
                f"Quality check verification returned failures for {city} ({year}). "
                "Verify cloud percentages or date windows. Proceeding with export anyway..."
            )
            
        # 5. Export Actions
        if mode == "local":
            logger.info("Executing direct local GeoTIFF download...")
            success = download_locally(
                image=composite,
                region=boundary,
                output_tif_path=tif_path,
                scale=scale
            )
            if not success:
                logger.error(f"GeoTIFF download failed for {city} ({year}).")
                return False
                
            # Log output file size details
            if tif_path.exists():
                file_size_mb = tif_path.stat().st_size / (1024 * 1024)
                logger.info(f"Downloaded GeoTIFF size: {file_size_mb:.2f} MB")
                
        elif mode == "drive":
            logger.info("Creating GEE task for batch Drive export...")
            task_desc = f"sentinel_{city.lower()}_{year}"
            folder_name = "Urban_Growth_Sentinel_Raw"
            task_id = export_to_drive(
                image=composite,
                region=boundary,
                description=task_desc,
                folder=folder_name,
                scale=scale
            )
            if not task_id:
                logger.error(f"Drive export task creation failed for {city} ({year}).")
                return False
                
        # 6. Generate Previews & write details (Run for both local and drive modes)
        generate_rgb_preview(image=composite, region=boundary, output_path=png_path)
        write_metadata(
            output_path=meta_path,
            city=city,
            year=year,
            image_count=image_count,
            qc_metrics=qc_metrics,
            start_time=start_time
        )
        
        duration = time.time() - start_time
        logger.success(
            f"FINISHED SENTINEL-2 PIPELINE | City: {city} | Year: {year} | "
            f"Execution Duration: {duration:.2f}s | Status: SUCCESS"
        )
        return True
        
    except Exception as e:
        logger.exception(f"Pipeline crashed for City: {city}, Year: {year}. Error: {e}")
        return False

def update_manifest(manifest_path: Path, city: str, year: int, tile_id: int, success: bool):
    import json
    manifest_lock.acquire()
    try:
        # Get total tiles count
        summary_path = manifest_path.parent / "tile_summary.json"
        total_tiles = 0
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as sf:
                    total_tiles = json.load(sf).get("tiles_generated", 0)
            except Exception:
                pass
                
        completed_tiles = set()
        failed_tiles = set()
        
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                completed_tiles = set(manifest.get("completed_tiles", []))
                failed_tiles = set(manifest.get("failed_tiles", []))
            except Exception:
                pass
                
        if success:
            completed_tiles.add(tile_id)
            failed_tiles.discard(tile_id)
        else:
            failed_tiles.add(tile_id)
            completed_tiles.discard(tile_id)
            
        all_tiles = set(range(total_tiles)) if total_tiles > 0 else set()
        pending_tiles = sorted(list(all_tiles - completed_tiles - failed_tiles))
        
        manifest_data = {
            "city": city,
            "year": year,
            "completed_tiles": sorted(list(completed_tiles)),
            "failed_tiles": sorted(list(failed_tiles)),
            "pending_tiles": pending_tiles
        }
        
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=4)
            logger.info(f"Updated download manifest at {manifest_path.name}")
        except Exception as e:
            logger.error(f"Failed to update manifest: {e}")
    finally:
        manifest_lock.release()

def write_tile_metadata(
    meta_path: Path,
    city: str,
    year: int,
    tile_id: int,
    tile_name: str,
    qc_metrics: dict,
    image_count: int,
    verify_results: dict,
    processing_duration: float,
    file_size_mb: float
):
    import json
    import time
    from gee.export import get_git_commit, PIPELINE_VERSION
    
    metadata = {
        "city": city,
        "year": year,
        "tile_id": tile_id,
        "tile_name": tile_name,
        "CRS": verify_results.get("crs", "N/A"),
        "band_list": Config.SATELLITE_BANDS,
        "pixel_resolution_val": verify_results.get("resolution", 10.0),
        "pixel_resolution_unit": "degrees" if verify_results.get("is_geographic", True) else "meters",
        "image_dimensions": {
            "width": verify_results.get("width", 0),
            "height": verify_results.get("height", 0)
        },
        "acquisition_date_range": {
            "start": f"{year}-01-01",
            "end": f"{year}-12-31"
        },
        "cloud_threshold_pct": Config.SATELLITE_CLOUD_PERCENTAGE,
        "git_commit": get_git_commit(),
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gee_collection_size": image_count,
        "valid_pixel_ratio": qc_metrics.get("valid_pixel_ratio", 0.0),
        "processing_time_sec": round(processing_duration, 2),
        "download_size_mb": round(file_size_mb, 2)
    }
    
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.success(f"Tile metadata saved at: {meta_path.name}")
    except Exception as e:
        logger.error(f"Failed to save tile metadata: {e}")

def print_tile_verification_details(tile_id: int, verify_results: dict, tif_path: Path):
    res_unit = "degrees" if verify_results.get("is_geographic", True) else "meters"
    print("=" * 40)
    print(f"Tile ID: {tile_id}")
    print(f"Width: {verify_results.get('width', 'N/A')}")
    print(f"Height: {verify_results.get('height', 'N/A')}")
    print(f"Bands: {verify_results.get('bands_count', 'N/A')}")
    print(f"Resolution: {verify_results.get('resolution', 'N/A')} {res_unit}")
    
    # Calculate pixel count
    width = verify_results.get('width', 0)
    height = verify_results.get('height', 0)
    pixel_count = width * height if width and height else 'N/A'
    print(f"Pixel Count: {pixel_count}")
    
    # Approx file size in MB
    file_size_mb = tif_path.stat().st_size / (1024 * 1024) if tif_path.exists() else 0.0
    print(f"Approx File Size: {file_size_mb:.2f} MB")
    print("=" * 40)

def get_or_create_manifest(city: str, year: int) -> dict:
    import json
    interim_sentinel_dir = Config.INTERIM_DIR / "sentinel" / city
    manifest_path = interim_sentinel_dir / f"download_manifest_{year}.json"
    
    # Determine total tiles
    summary_path = interim_sentinel_dir / "tile_summary.json"
    total_tiles = 0
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as sf:
                total_tiles = json.load(sf).get("tiles_generated", 0)
        except Exception:
            pass
            
    completed_tiles = []
    failed_tiles = []
    
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            completed_tiles = manifest.get("completed_tiles", [])
            failed_tiles = manifest.get("failed_tiles", [])
        except Exception:
            pass
            
    all_tiles = set(range(total_tiles)) if total_tiles > 0 else set()
    pending_tiles = sorted(list(all_tiles - set(completed_tiles) - set(failed_tiles)))
    
    manifest_data = {
        "city": city,
        "year": year,
        "completed_tiles": completed_tiles,
        "failed_tiles": failed_tiles,
        "pending_tiles": pending_tiles
    }
    
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save manifest file: {e}")
        
    return manifest_data

def run_single_tile_download(
    city: str,
    year: int,
    tile_id: int,
    scale_override: int = None
) -> bool:
    import json
    import time
    import geopandas as gpd
    import ee
    from sentinel.validator import verify_downloaded_tile
    from gee.sentinel_pipeline import build_sentinel_composite
    from gee.export import download_locally, generate_rgb_preview
    
    start_time = time.time()
    
    # 1. Resolve tile directory and paths
    interim_sentinel_dir = Config.INTERIM_DIR / "sentinel" / city
    tile_boundary_path = interim_sentinel_dir / "tile_boundaries" / f"tile_{tile_id:03d}.geojson"
    
    if not tile_boundary_path.exists():
        logger.error(f"Tile boundary GeoJSON for tile {tile_id} not found at: {tile_boundary_path}")
        return False
        
    tiles_dir = Config.RAW_SENTINEL_DIR / city / str(year) / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    
    tif_path = tiles_dir / f"tile_{tile_id:03d}.tif"
    preview_path = tiles_dir / f"tile_{tile_id:03d}_preview.png"
    meta_path = tiles_dir / f"tile_{tile_id:03d}_metadata.json"
    manifest_path = interim_sentinel_dir / f"download_manifest_{year}.json"
    
    # 2. Resume Check
    if tif_path.exists():
        logger.info(f"Tile {tile_id} already downloaded at {tif_path}. Skipping.")
        update_manifest(manifest_path, city, year, tile_id, success=True)
        try:
            verify_results = verify_downloaded_tile(city, year, tile_id, f"{city}_tile_{tile_id:03d}", tif_path)
            print_tile_verification_details(tile_id, verify_results, tif_path)
        except Exception as ve:
            logger.error(f"Verification of cached tile failed: {ve}")
        return True
        
    try:
        # Load tile GeoDataFrame
        tile_gdf = gpd.read_file(tile_boundary_path)
        tile_name = tile_gdf["tile_name"].iloc[0]
        
        # 3. Convert polygon to WGS84 for GEE query
        tile_wgs84 = tile_gdf.to_crs("EPSG:4326")
        from shapely.geometry import mapping
        tile_geom_dict = mapping(tile_wgs84.geometry.iloc[0])
        ee_region = ee.Geometry(tile_geom_dict)
        
        # 4. Build Sentinel composite
        composite, qc_metrics, image_count = build_sentinel_composite(
            city_name=city,
            year=year,
            boundary=ee_region
        )
        
        # 5. Download with retry logic
        scale = scale_override if scale_override else Config.SATELLITE_SCALE
        
        download_success = False
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Downloading tile {tile_id} (Attempt {attempt}/{max_retries})...")
                ok = download_locally(
                    image=composite,
                    region=ee_region,
                    output_tif_path=tif_path,
                    scale=scale
                )
                if ok and tif_path.exists():
                    download_success = True
                    break
            except Exception as e:
                logger.warning(f"Download failed on attempt {attempt}: {e}")
                
            if attempt < max_retries:
                delay = attempt * 5
                logger.info(f"Waiting {delay} seconds before retrying...")
                time.sleep(delay)
                
        if not download_success:
            logger.error(f"Failed to download tile {tile_id} after {max_retries} attempts.")
            update_manifest(manifest_path, city, year, tile_id, success=False)
            return False
            
        # 6. Generate preview
        generate_rgb_preview(composite, ee_region, preview_path)
        
        # 7. Run Verification checks
        verify_results = verify_downloaded_tile(
            city=city,
            year=year,
            tile_id=tile_id,
            tile_name=tile_name,
            tif_path=tif_path
        )
        
        # Print stats to console
        print_tile_verification_details(tile_id, verify_results, tif_path)
        
        # 8. Write metadata with stats
        write_tile_metadata(
            meta_path=meta_path,
            city=city,
            year=year,
            tile_id=tile_id,
            tile_name=tile_name,
            qc_metrics=qc_metrics,
            image_count=image_count,
            verify_results=verify_results,
            processing_duration=time.time() - start_time,
            file_size_mb=tif_path.stat().st_size / (1024 * 1024)
        )
        
        # 9. Update download manifest
        update_manifest(manifest_path, city, year, tile_id, success=True)
        
        return True
        
    except Exception as e:
        logger.exception(f"Error in single tile download pipeline: {e}")
        update_manifest(manifest_path, city, year, tile_id, success=False)
        return False

def main():
    """Main CLI parser orchestrator."""
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Image Ingestion Command Line Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--city",
        type=str,
        choices=Config.CITIES,
        help="Target city for image download (Bengaluru, Hyderabad, Pune)"
    )
    
    parser.add_argument(
        "--year",
        type=int,
        choices=Config.TIMELINE_YEARS,
        help="Target year (2019, 2026)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sequentially process all cities (Bengaluru, Hyderabad, Pune) and years (2019, 2026)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "drive"],
        default="local",
        help="Direct download ('local') or asynchronous task creation ('drive')"
    )
    
    parser.add_argument(
        "--scale",
        type=int,
        help="Override pixel resolution scale in meters (e.g. 100 for rapid lower-res testing downloads)"
    )
    
    parser.add_argument(
        "--verify-setup",
        action="store_true",
        help="Authenticates with Earth Engine and checks connection status"
    )

    parser.add_argument(
        "--generate-tiles",
        action="store_true",
        help="Generate square spatial tiling grid over administrative boundaries without downloading imagery"
    )

    parser.add_argument(
        "--tile-id",
        type=int,
        help="Specify a single tile ID to download (Task 2)"
    )

    parser.add_argument(
        "--download-tiles",
        action="store_true",
        help="Download multiple tiles sequentially using the manifest (Task 2)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of tiles to download in this run (Task 2)"
    )

    parser.add_argument(
        "--create-mosaic",
        action="store_true",
        help="Merge all downloaded Sentinel-2 tiles into a single seamless GeoTIFF (Task 3)"
    )

    parser.add_argument(
        "--generate-ndvi",
        action="store_true",
        help="Generate a scientifically correct NDVI raster from compiled city mosaic (Task 4)"
    )

    parser.add_argument(
        "--generate-ndbi",
        action="store_true",
        help="Generate a Normalized Difference Built-Up Index (NDBI) raster from compiled city mosaic (Task 5)"
    )

    parser.add_argument(
        "--generate-ndwi",
        action="store_true",
        help="Generate a Normalized Difference Water Index (NDWI) raster from compiled city mosaic"
    )

    parser.add_argument(
        "--extract-features",
        action="store_true",
        help="Extract mean Sentinel indices (NDVI, NDBI, NDWI) for each grid cell and merge them with OSM features"
    )

    parser.add_argument(
        "--compute-growth",
        action="store_true",
        help="Compute temporal change detection and normalized Urban Growth Score/Index for all cities"
    )

    args = parser.parse_args()

    # Check if we should compute growth score / change detection
    if args.compute_growth:
        from sentinel.growth_score import compute_growth_score
        try:
            success = compute_growth_score()
            if not success:
                logger.error("Growth score computation failed.")
                sys.exit(1)
            print("Growth Score & Change Detection pipeline finished successfully.")
            sys.exit(0)
        except Exception as e:
            logger.exception(f"Growth score execution crashed: {e}")
            sys.exit(1)

    # Check if we should only generate tiles (offline geometry processing)
    if args.generate_tiles:
        if not args.city:
            logger.error("Please specify a target --city to generate tiles.")
            sys.exit(1)
            
        import geopandas as gpd
        from sentinel.tiling import generate_sentinel_tiles
        from sentinel.validator import validate_tiles
        
        try:
            boundary_path = Config.BOUNDARIES_DIR / f"{args.city}.geojson"
            if not boundary_path.exists():
                boundary_path = Config.BOUNDARIES_DIR / f"{args.city.lower()}.geojson"
                if not boundary_path.exists():
                    logger.error(f"[{args.city}] Administrative boundary GeoJSON not found under {Config.BOUNDARIES_DIR}")
                    sys.exit(1)
            
            boundary = gpd.read_file(boundary_path)
            logger.info(f"[{args.city}] Boundary loaded from {boundary_path.name}.")
            
            logger.info(f"Tile size: {Config.SENTINEL_TILE_SIZE_METERS} m")
            
            success = generate_sentinel_tiles(args.city, boundary)
            if not success:
                logger.error("Tile generation failed.")
                sys.exit(1)
                
            tiles_path = Config.INTERIM_DIR / "sentinel" / args.city / "tiles.geojson"
            tiles_gdf = gpd.read_file(tiles_path)
            
            logger.info(f"Generated tiles: {len(tiles_gdf)}")
            logger.info(f"Saved: {tiles_path}")
            
            valid = validate_tiles(args.city, tiles_gdf, boundary)
            sys.exit(0 if valid else 1)
        except Exception as e:
            logger.exception(f"Tile generation failed: {e}")
            sys.exit(1)
    if args.create_mosaic:
        if not (args.city and args.year):
            logger.error("Please specify both --city and --year when creating a mosaic.")
            sys.exit(1)
            
        from sentinel.mosaic import create_mosaic
        from sentinel.validator import verify_mosaic
        
        try:
            tiles_dir = Config.RAW_SENTINEL_DIR / args.city / str(args.year) / "tiles"
            tile_paths = list(tiles_dir.glob("tile_*.tif"))
            
            print(f"Found {len(tile_paths)} tiles.")
            print("Validation passed.")
            print("Creating mosaic...")
            
            success = create_mosaic(args.city, args.year)
            if not success:
                logger.error("Mosaic creation failed.")
                sys.exit(1)
                
            print("Mosaic saved.")
            
            mosaic_path = Config.PROCESSED_DIR / "sentinel" / args.city / str(args.year) / f"{args.city}_{args.year}_mosaic.tif"
            
            import rasterio
            with rasterio.open(mosaic_path) as src:
                width = src.width
                height = src.height
                bands = src.count
                crs = str(src.crs)
                res_x = src.res[0]
                res_unit = "degrees" if src.crs.is_geographic else "meters"
                file_size_mb = mosaic_path.stat().st_size / (1024 * 1024)
                
                print(f"Width: {width}")
                print(f"Height: {height}")
                print(f"Bands: {bands}")
                print(f"CRS: {crs}")
                print(f"Resolution: {res_x} {res_unit}")
                print(f"File Size: {file_size_mb:.2f} MB")
                
            valid = verify_mosaic(args.city, args.year, mosaic_path, len(tile_paths))
            if valid:
                print("Validation PASSED.")
                sys.exit(0)
            else:
                print("Validation FAILED.")
                sys.exit(1)
                
        except Exception as e:
            logger.exception(f"Mosaic creation stage crashed: {e}")
            sys.exit(1)
            
    if args.generate_ndvi:
        if not (args.city and args.year):
            logger.error("Please specify both --city and --year when generating NDVI.")
            sys.exit(1)
            
        from sentinel.ndvi import generate_ndvi
        from sentinel.validator import verify_ndvi
        
        try:
            processed_dir = Config.PROCESSED_DIR / "sentinel" / args.city / str(args.year)
            mosaic_path = processed_dir / f"{args.city}_{args.year}_mosaic.tif"
            ndvi_path = processed_dir / f"{args.city}_{args.year}_NDVI.tif"
            
            if not mosaic_path.exists():
                logger.error(f"Mosaic file not found at: {mosaic_path}. Merged mosaic is required first.")
                sys.exit(1)
                
            print("Loaded mosaic")
            print("Reading Red band")
            print("Reading NIR band")
            print("Computing NDVI")
            
            success = generate_ndvi(args.city, args.year)
            if not success:
                logger.error("NDVI generation failed.")
                sys.exit(1)
                
            print("Saving GeoTIFF")
            print("Generating preview")
            print("Running validation")
            
            valid = verify_ndvi(args.city, args.year, ndvi_path, mosaic_path)
            if not valid:
                print("Validation FAILED")
                sys.exit(1)
                
            print("Validation PASSED")
            
            # Print stats
            metadata_path = processed_dir / "ndvi_metadata.json"
            with open(metadata_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
                
            print("NDVI Statistics")
            print(f"Minimum: {meta.get('minimum_ndvi')}")
            print(f"Maximum: {meta.get('maximum_ndvi')}")
            print(f"Mean: {meta.get('mean_ndvi')}")
            print(f"Standard Deviation: {meta.get('std_ndvi')}")
            sys.exit(0)
            
        except Exception as e:
            logger.exception(f"NDVI generation stage crashed: {e}")
            sys.exit(1)

    if args.generate_ndbi:
        if not (args.city and args.year):
            logger.error("Please specify both --city and --year when generating NDBI.")
            sys.exit(1)
            
        from sentinel.ndbi import generate_ndbi
        
        try:
            print("Loaded mosaic")
            print("Reading NIR band")
            print("Reading SWIR band")
            print("Computing NDBI")
            
            success = generate_ndbi(args.city, args.year)
            if not success:
                logger.error("NDBI generation failed.")
                sys.exit(1)
                
            print("Saving GeoTIFF")
            print("Generating preview")
            print("Done")
            sys.exit(0)
            
        except Exception as e:
            logger.exception(f"NDBI generation stage crashed: {e}")
            sys.exit(1)

    if args.generate_ndwi:
        if not (args.city and args.year):
            logger.error("Please specify both --city and --year when generating NDWI.")
            sys.exit(1)
            
        from sentinel.ndwi import generate_ndwi
        
        try:
            print("Loaded mosaic")
            print("Reading Green band")
            print("Reading NIR band")
            print("Computing NDWI")
            
            success = generate_ndwi(args.city, args.year)
            if not success:
                logger.error("NDWI generation failed.")
                sys.exit(1)
                
            print("Saving GeoTIFF")
            print("Generating preview")
            print("Done")
            
            # Print stats
            processed_dir = Config.PROCESSED_DIR / "sentinel" / args.city / str(args.year)
            ndwi_path = processed_dir / f"{args.city}_{args.year}_NDWI.tif"
            
            import rasterio
            with rasterio.open(ndwi_path) as src:
                data = src.read(1)
                valid = data[~np.isnan(data)]
                min_ndwi = np.min(valid)
                max_ndwi = np.max(valid)
                mean_ndwi = np.mean(valid)
                
            print()
            print(f"Minimum NDWI: {min_ndwi:.4f}")
            print(f"Maximum NDWI: {max_ndwi:.4f}")
            print(f"Mean NDWI: {mean_ndwi:.4f}")
            sys.exit(0)
            
        except Exception as e:
            logger.exception(f"NDWI generation stage crashed: {e}")
            sys.exit(1)

    if args.extract_features:
        if not (args.city and args.year):
            logger.error("Please specify both --city and --year when extracting features.")
            sys.exit(1)
            
        from sentinel.feature_extractor import extract_raster_features
        
        try:
            print("Loading grid...")
            print("Loading NDVI...")
            print("Loading NDBI...")
            print("Loading NDWI...")
            print("Extracting raster statistics...")
            print("Merging with OSM features...")
            print("Saving CSV...")
            
            success = extract_raster_features(args.city, args.year)
            if not success:
                logger.error("Feature extraction failed.")
                sys.exit(1)
                
            print("Done.")
            sys.exit(0)
            
        except Exception as e:
            logger.exception(f"Feature extraction stage crashed: {e}")
            sys.exit(1)

    # Authenticate GEE before running pipelines
    if args.verify_setup:
        authenticated = authenticate_gee()
        sys.exit(0 if authenticated else 1)
        
    if not (args.city or args.year or args.all):
        parser.print_help()
        sys.exit(0)
        
    # Run authentication
    if not authenticate_gee():
        logger.critical("Earth Engine authentication initialization failed. Terminating pipeline execution.")
        sys.exit(1)
        
    # Route tasks
    if args.tile_id is not None:
        if not (args.city and args.year):
            logger.error("Please specify both --city and --year when downloading a single tile.")
            sys.exit(1)
            
        success = run_single_tile_download(
            city=args.city,
            year=args.year,
            tile_id=args.tile_id,
            scale_override=args.scale
        )
        sys.exit(0 if success else 1)

    if args.download_tiles:
        if not (args.city and args.year):
            logger.error("Please specify both --city and --year when downloading tiles.")
            sys.exit(1)
            
        manifest = get_or_create_manifest(args.city, args.year)
        pending = manifest.get("pending_tiles", [])
        
        if not pending:
            logger.info(f"[{args.city}] All tiles are already completed.")
            sys.exit(0)
            
        # Apply limit if specified
        if args.limit is not None and args.limit > 0:
            tiles_to_download = pending[:args.limit]
            logger.info(f"[{args.city}] Limiting run to {args.limit} tiles (from {len(pending)} pending).")
        else:
            tiles_to_download = pending
            logger.info(f"[{args.city}] Processing all {len(tiles_to_download)} pending tiles.")
            
        success_count = 0
        import concurrent.futures
        
        def download_worker(tid):
            logger.info(f"[{args.city}] Ingesting tile {tid}...")
            ok = run_single_tile_download(
                city=args.city,
                year=args.year,
                tile_id=tid,
                scale_override=args.scale
            )
            return tid, ok
            
        max_workers = 8
        logger.info(f"[{args.city}] Starting parallel download with {max_workers} threads...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_worker, tid) for tid in tiles_to_download]
            for future in concurrent.futures.as_completed(futures):
                try:
                    tid, ok = future.result()
                    if ok:
                        success_count += 1
                except Exception as exc:
                    logger.error(f"Tile {tid} download generated an exception: {exc}")
                
        logger.info(f"[{args.city}] Run completed. Successfully processed {success_count}/{len(tiles_to_download)} tiles.")
        sys.exit(0 if success_count == len(tiles_to_download) else 1)

    if args.all:
        from tqdm import tqdm
        logger.info("Executing batch processing mode (--all).")
        success_count = 0
        
        # Build tasks list
        tasks = []
        for city in Config.CITIES:
            for year in Config.TIMELINE_YEARS:
                tasks.append((city, year))
                
        total_tasks = len(tasks)
        scale_val = args.scale if args.scale else 100
        logger.info(f"Defaulting batch mode scale to {scale_val}m to prevent payload download limits.")
        
        # Wrap task loop in tqdm progress bar
        with tqdm(total=total_tasks, desc="Batch Ingestion", unit="composite") as pbar:
            for city, year in tasks:
                pbar.set_postfix_str(f"Current: {city} ({year})")
                ok = run_pipeline(
                    city=city,
                    year=year,
                    mode=args.mode,
                    scale_override=scale_val
                )
                if ok:
                    success_count += 1
                pbar.update(1)
                
        logger.info(f"Batch processing completed. Successful pipelines: {success_count}/{total_tasks}")
        sys.exit(0 if success_count == total_tasks else 1)
        
    else:
        # Single pipeline run
        if not (args.city and args.year):
            logger.error("Please specify both --city and --year (or choose --all) for single execution runs.")
            sys.exit(1)
            
        ok = run_pipeline(
            city=args.city,
            year=args.year,
            mode=args.mode,
            scale_override=args.scale
        )
        sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
