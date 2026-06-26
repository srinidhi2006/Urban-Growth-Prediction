"""
Sentinel-2 Image Ingestion Orchestrator.
CLI script to query, compose, quality-check, and export satellite layers.
"""

import argparse
import sys
import time
from pathlib import Path
from loguru import logger

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

    args = parser.parse_args()
    
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
