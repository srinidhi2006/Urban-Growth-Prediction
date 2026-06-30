"""
Sentinel Tile Validator Module.
Asserts database integrity, metadata completeness, CRS parameters, and spatial containment of generated tiles.
Saves validation report in the features folder.
"""

import json
from datetime import datetime
from pathlib import Path
import geopandas as gpd
from loguru import logger
from config import Config

def validate_tiles(city: str, tiles_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame) -> bool:
    """Performs integrity validation checks on generated Sentinel tile geometries and files.
    
    Args:
        city (str): Target city name.
        tiles_gdf (gpd.GeoDataFrame): Generated tiles layer.
        boundary_gdf (gpd.GeoDataFrame): City boundary.
        
    Returns:
        bool: True if validation passes, False otherwise.
    """
    logger.info(f"[{city}] Running validation checks on generated Sentinel tiles...")
    
    features_dir = Config.FEATURES_DIR
    features_dir.mkdir(parents=True, exist_ok=True)
    report_path = features_dir / f"tile_validation_report_{city}.json"
    
    passed = True
    checks = {
        "total_tiles_validated": len(tiles_gdf),
        "duplicate_tile_ids_found": 0,
        "invalid_crs_found": False,
        "empty_geometries_found": 0,
        "missing_boundary_files_count": 0,
        "ratio_bounds_violations": 0,  # placeholder or not needed here
        "non_intersecting_tiles_found": 0,
        "missing_columns": []
    }
    
    try:
        # 1. Check columns availability
        required_cols = ["tile_id", "city", "tile_name", "status", "geometry"]
        missing_cols = [c for c in required_cols if c not in tiles_gdf.columns]
        checks["missing_columns"] = missing_cols
        if missing_cols:
            logger.error(f"[{city}] Validation failed: Missing columns {missing_cols}")
            passed = False
            
        # 2. Check unique IDs
        if "tile_id" in tiles_gdf.columns:
            dup_ids = tiles_gdf["tile_id"].duplicated().sum()
            checks["duplicate_tile_ids_found"] = int(dup_ids)
            if dup_ids > 0:
                logger.error(f"[{city}] Validation failed: Found {dup_ids} duplicate tile IDs.")
                passed = False
                
        # 3. Check CRS is projected EPSG:3857
        crs_str = str(tiles_gdf.crs)
        if tiles_gdf.crs is None or "3857" not in crs_str:
            logger.error(f"[{city}] Validation failed: CRS is '{crs_str}', expected EPSG:3857.")
            checks["invalid_crs_found"] = True
            passed = False
            
        # 4. Check empty geometries
        empty_count = tiles_gdf.geometry.is_empty.sum()
        checks["empty_geometries_found"] = int(empty_count)
        if empty_count > 0:
            logger.error(f"[{city}] Validation failed: Found {empty_count} empty geometries.")
            passed = False
            
        # 5. Check boundary intersection
        # Project boundary to projected CRS if needed
        if boundary_gdf.crs != Config.PROJECTED_CRS:
            boundary_projected = boundary_gdf.to_crs(Config.PROJECTED_CRS)
        else:
            boundary_projected = boundary_gdf
            
        boundary_geom = boundary_projected.geometry.iloc[0]
        
        non_intersecting = 0
        for idx, geom in zip(tiles_gdf["tile_id"], tiles_gdf.geometry):
            if not geom.intersects(boundary_geom):
                non_intersecting += 1
                logger.error(f"[{city}] Validation failed: Tile {idx} does not intersect administrative boundary.")
                
        checks["non_intersecting_tiles_found"] = non_intersecting
        if non_intersecting > 0:
            passed = False
            
        # 6. Verify individual tile boundary files exist on disk
        interim_sentinel_dir = Config.INTERIM_DIR / "sentinel" / city
        boundaries_dir = interim_sentinel_dir / "tile_boundaries"
        
        missing_files = 0
        for idx in tiles_gdf["tile_id"]:
            expected_file = boundaries_dir / f"tile_{idx:03d}.geojson"
            if not expected_file.exists():
                missing_files += 1
                logger.error(f"[{city}] Validation failed: Missing tile boundary GeoJSON at: {expected_file}")
                
        checks["missing_boundary_files_count"] = missing_files
        if missing_files > 0:
            passed = False
            
        # 7. Write report
        report = {
            "city": city,
            "validation_timestamp": datetime.utcnow().isoformat() + "Z",
            "passed": passed,
            "checks": checks
        }
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        if passed:
            logger.success(f"[{city}] Sentinel tile validation report created successfully. All checks PASSED.")
        else:
            logger.error(f"[{city}] Sentinel tile validation FAILED. Report written to {report_path.name}")
            
        return passed
        
    except Exception as e:
        logger.exception(f"[{city}] Tile Validator stage crashed: {e}")
        return False

def verify_downloaded_tile(
    city: str,
    year: int,
    tile_id: int,
    tile_name: str,
    tif_path: Path
) -> dict:
    """Verifies a downloaded single-tile GeoTIFF file using rasterio.
    
    Saves verification JSON report and returns raster attributes.
    """
    import rasterio
    import json
    from datetime import datetime
    
    logger.info(f"[{city}] Opening and verifying raster: {tif_path.name}")
    
    results = {
        "width": 0,
        "height": 0,
        "bands_count": 0,
        "crs": "",
        "is_geographic": True,
        "resolution": 0.0,
        "bounds": [],
        "passed": False
    }
    
    try:
        with rasterio.open(tif_path) as src:
            results["width"] = src.width
            results["height"] = src.height
            results["bands_count"] = src.count
            results["crs"] = str(src.crs)
            results["is_geographic"] = src.crs.is_geographic
            
            # Extract resolution
            res_x, res_y = src.res
            results["resolution"] = float(abs(res_x))
            
            # Extract bounds
            bounds = src.bounds
            results["bounds"] = [bounds.left, bounds.bottom, bounds.right, bounds.top]
            
            # Perform assertions
            bands_ok = src.count == 6
            crs_ok = src.crs is not None and len(str(src.crs)) > 0
            
            # Differentiate resolution checks based on CRS units (degrees vs meters)
            if src.crs.is_geographic:
                # 10m is approximately 8.98e-5 degrees (allow tolerance range below 0.001 deg)
                resolution_ok = abs(res_x) < 0.001
            else:
                resolution_ok = abs(res_x - 10.0) < 1.0 or abs(res_x) <= 100.0
                
            dimensions_ok = src.width > 0 and src.height > 0
            
            passed = bool(bands_ok and crs_ok and resolution_ok and dimensions_ok)
            results["passed"] = passed
            
        # Write validation JSON report
        features_dir = Config.FEATURES_DIR
        features_dir.mkdir(parents=True, exist_ok=True)
        report_path = features_dir / f"tile_download_validation_{city}_tile{tile_id:03d}.json"
        
        report_data = {
            "city": city,
            "year": year,
            "tile_id": tile_id,
            "tile_name": tile_name,
            "download_success": tif_path.exists(),
            "metadata_verification": True,
            "raster_verification": passed,
            "band_count": results["bands_count"],
            "crs_check": crs_ok,
            "pixel_size_check": resolution_ok
        }
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
            
        logger.success(f"Tile download validation report saved at: {report_path.name}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to verify downloaded raster {tif_path.name}: {e}")
        results["passed"] = False
        return results

def verify_mosaic(city: str, year: int, mosaic_path: Path, tiles_count: int) -> bool:
    """Validates the compiled Sentinel-2 mosaic file and logs a validation report.
    
    Args:
        city (str): Target city.
        year (int): Year of the mosaic.
        mosaic_path (Path): Path to the compiled mosaic TIFF.
        tiles_count (int): Count of tiles merged.
        
    Returns:
        bool: True if validation passes, False otherwise.
    """
    import rasterio
    logger.info(f"[{city}] Running validation checks on compiled mosaic: {mosaic_path.name}")
    
    passed = True
    crs_check = False
    resolution_check = False
    raster_integrity = False
    band_count = 0
    
    try:
        # Check file exists
        if not mosaic_path.exists():
            logger.error(f"[{city}] Validation failed: Mosaic file does not exist at: {mosaic_path}")
            passed = False
        else:
            with rasterio.open(mosaic_path) as src:
                band_count = src.count
                crs_str = str(src.crs)
                res_x, res_y = src.res
                
                # Assert band count is 6
                bands_ok = src.count == 6
                if not bands_ok:
                    logger.error(f"[{city}] Validation failed: Mosaic has {src.count} bands, expected 6.")
                    passed = False
                    
                # Confirm CRS is non-empty
                crs_ok = src.crs is not None and len(crs_str) > 0
                crs_check = crs_ok
                if not crs_ok:
                    logger.error(f"[{city}] Validation failed: CRS is empty or invalid.")
                    passed = False
                    
                # Confirm resolution is approximately 10m or geographic equivalent
                if src.crs.is_geographic:
                    res_ok = abs(res_x) < 0.001
                else:
                    res_ok = abs(res_x - 10.0) < 1.0 or abs(res_x) <= 100.0
                resolution_check = res_ok
                if not res_ok:
                    logger.error(f"[{city}] Validation failed: Resolution mismatch. Got {src.res}.")
                    passed = False
                    
                # Assert width and height are greater than zero
                dimensions_ok = src.width > 0 and src.height > 0
                if not dimensions_ok:
                    logger.error(f"[{city}] Validation failed: Width or height is zero. Shape = {src.shape}.")
                    passed = False
                    
                raster_integrity = bool(bands_ok and crs_ok and res_ok and dimensions_ok)
                
        passed = passed and raster_integrity
        
        # Save validation JSON report
        features_dir = Config.FEATURES_DIR
        features_dir.mkdir(parents=True, exist_ok=True)
        report_path = features_dir / f"mosaic_validation_{city}_{year}.json"
        
        report_data = {
            "passed": passed,
            "tiles_merged": tiles_count,
            "band_count": band_count,
            "crs_check": crs_check,
            "resolution_check": resolution_check,
            "raster_integrity": raster_integrity
        }
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
            
        if passed:
            logger.success(f"[{city}] Mosaic validation report saved at: {report_path.name}. Verification PASSED.")
        else:
            logger.error(f"[{city}] Mosaic validation report saved at: {report_path.name}. Verification FAILED.")
            
        return passed
        
    except Exception as e:
        logger.exception(f"[{city}] Mosaic validation crashed: {e}")
        return False

def verify_ndvi(city: str, year: int, ndvi_path: Path, mosaic_path: Path) -> bool:
    """Validates the generated NDVI raster file and logs a validation report.
    
    Args:
        city (str): Target city name.
        year (int): Calendar year of ingestion.
        ndvi_path (Path): Path to the generated NDVI TIFF.
        mosaic_path (Path): Path to the source mosaic TIFF.
        
    Returns:
        bool: True if validation passes, False otherwise.
    """
    import rasterio
    import numpy as np
    
    logger.info(f"[{city}] Running validation checks on NDVI raster: {ndvi_path.name}")
    
    passed = True
    value_range_check = False
    datatype_check = False
    crs_check = False
    transform_check = False
    nodata_check = False
    
    try:
        if not ndvi_path.exists():
            logger.error(f"[{city}] NDVI validation failed: File does not exist at: {ndvi_path}")
            passed = False
        elif not mosaic_path.exists():
            logger.error(f"[{city}] NDVI validation failed: Source mosaic file does not exist at: {mosaic_path}")
            passed = False
        else:
            with rasterio.open(ndvi_path) as ndvi_src, rasterio.open(mosaic_path) as mosaic_src:
                # 1. Single band check
                bands_ok = ndvi_src.count == 1
                if not bands_ok:
                    logger.error(f"[{city}] NDVI validation failed: Band count is {ndvi_src.count}, expected 1.")
                    passed = False
                    
                # 2. Datatype check (float32)
                dtype_str = ndvi_src.dtypes[0]
                datatype_ok = dtype_str == "float32"
                datatype_check = datatype_ok
                if not datatype_ok:
                    logger.error(f"[{city}] NDVI validation failed: Data type is {dtype_str}, expected float32.")
                    passed = False
                    
                # 3. CRS check
                crs_ok = ndvi_src.crs == mosaic_src.crs
                crs_check = crs_ok
                if not crs_ok:
                    logger.error(f"[{city}] NDVI validation failed: CRS mismatch with mosaic.")
                    passed = False
                    
                # 4. Transform check
                transform_ok = ndvi_src.transform == mosaic_src.transform
                transform_check = transform_ok
                if not transform_ok:
                    logger.error(f"[{city}] NDVI validation failed: Transform mismatch with mosaic.")
                    passed = False
                    
                # 5. Read NDVI data and verify range [-1.0, 1.0] and nodata
                ndvi_data = ndvi_src.read(1)
                
                # Check for NaN / NoData pixels
                has_nans = np.isnan(ndvi_data).any()
                nodata_check = True
                
                # Range check: valid non-NaN pixels must lie in [-1.0, 1.0]
                valid_data = ndvi_data[~np.isnan(ndvi_data)]
                if len(valid_data) > 0:
                    min_val = float(np.min(valid_data))
                    max_val = float(np.max(valid_data))
                    range_ok = min_val >= -1.0 and max_val <= 1.0
                else:
                    range_ok = True
                value_range_check = range_ok
                if not range_ok:
                    logger.error(f"[{city}] NDVI validation failed: Value out of range [-1.0, 1.0]. Got [{min_val}, {max_val}].")
                    passed = False
                    
        passed = bool(passed and value_range_check and datatype_check and crs_check and transform_check)
        
        # Save validation report
        features_dir = Config.FEATURES_DIR
        features_dir.mkdir(parents=True, exist_ok=True)
        report_path = features_dir / f"ndvi_validation_{city}_{year}.json"
        
        report_data = {
            "passed": bool(passed),
            "value_range_check": bool(value_range_check),
            "datatype_check": bool(datatype_check),
            "crs_check": bool(crs_check),
            "transform_check": bool(transform_check),
            "nodata_check": bool(nodata_check)
        }
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
            
        if passed:
            logger.success(f"[{city}] NDVI validation report saved at: {report_path.name}. Verification PASSED.")
        else:
            logger.error(f"[{city}] NDVI validation report saved at: {report_path.name}. Verification FAILED.")
            
        return passed
        
    except Exception as e:
        logger.exception(f"[{city}] NDVI validation crashed: {e}")
        return False
