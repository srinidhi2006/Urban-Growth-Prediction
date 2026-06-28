"""
OSM Pipeline Orchestrator CLI.
Allows running the full OpenStreetMap download, cleaning, spatial join, and extraction workflow
for a single city, all target cities, or executing diagnostic checks.
"""

import argparse
import sys
import time
from pathlib import Path
import geopandas as gpd
from loguru import logger

# Add root folder to search path if executing directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config
from osm.downloader import download_osm_data
from osm.cleaner import clean_osm_data
from osm.grid_generator import generate_spatial_grid
from osm.spatial_join import join_spatial_data
from osm.feature_extractor import extract_osm_features
from osm.validator import validate_osm_features

def verify_setup() -> bool:
    """Verifies that required dependencies are importable and boundary geojsons exist.
    
    Returns:
        bool: True if verification passes, False otherwise.
    """
    logger.info("Verifying OSM processing pipeline environment and setup...")
    success = True
    
    # 1. Check dependencies
    dependencies = [
        ("osmnx", "OSMnx"),
        ("geopandas", "GeoPandas"),
        ("shapely", "Shapely"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy")
    ]
    
    logger.info("--- Dependency Verification ---")
    for module_name, name in dependencies:
        try:
            __import__(module_name)
            logger.info(f" [PASS] {name} is available and importable.")
        except ImportError:
            logger.error(f" [FAIL] {name} ({module_name}) is missing in the environment.")
            success = False
            
    # 2. Check boundaries
    logger.info("--- City Boundary Files Verification ---")
    for city in Config.CITIES:
        geojson_path = Config.BOUNDARIES_DIR / f"{city}.geojson"
        if geojson_path.exists():
            logger.info(f" [PASS] Boundary GeoJSON for {city} exists at: {geojson_path.name}")
        else:
            logger.error(f" [FAIL] Boundary GeoJSON for {city} is missing at: {geojson_path}")
            success = False
            
    if success:
        logger.success("OSM environment verification completed successfully.")
    else:
        logger.error("OSM environment verification failed. Please inspect requirements.")
        
    return success

def run_osm_pipeline(city: str, grid_size_override: int = None) -> bool:
    """Runs the OSM processing pipeline from download to extraction and validation.
    
    Args:
        city (str): Target city name.
        grid_size_override (int, optional): Grid cell resolution in meters.
        
    Returns:
        bool: True if pipeline completes successfully, False otherwise.
    """
    logger.info(f"======================================================================")
    logger.info(f"STARTING OSM PIPELINE | City: {city} | Grid Size: {grid_size_override or Config.OSM_GRID_SIZE_METERS}m")
    logger.info(f"======================================================================")
    
    start_time = time.time()
    
    # 1. Load boundary GeoDataFrame
    boundary_path = Config.BOUNDARIES_DIR / f"{city}.geojson"
    if not boundary_path.exists():
        logger.error(f"[{city}] Pipeline aborted: Boundary file missing at {boundary_path}")
        return False
        
    try:
        boundary_gdf = gpd.read_file(boundary_path)
        
        # Stage 1: Downloader
        if not download_osm_data(city, boundary_gdf):
            logger.error(f"[{city}] Downloader stage failed.")
            return False
            
        # Stage 2: Cleaner
        if not clean_osm_data(city, boundary_gdf):
            logger.error(f"[{city}] Cleaner stage failed.")
            return False
            
        # Stage 3: Grid Generator
        if not generate_spatial_grid(city, boundary_gdf, grid_size_meters=grid_size_override):
            logger.error(f"[{city}] Grid Generator stage failed.")
            return False
            
        # Stage 4: Spatial Join
        if not join_spatial_data(city):
            logger.error(f"[{city}] Spatial Join stage failed.")
            return False
            
        # Stage 5: Feature Extraction
        if not extract_osm_features(city, boundary_gdf):
            logger.error(f"[{city}] Feature Extraction stage failed.")
            return False
            
        # Stage 6: Validator
        if not validate_osm_features(city):
            logger.error(f"[{city}] Validation check reported warnings or errors.")
            return False
            
        duration = time.time() - start_time
        logger.success(
            f"FINISHED OSM PIPELINE | City: {city} | "
            f"Execution Duration: {duration:.2f}s | Status: SUCCESS"
        )
        return True
        
    except Exception as e:
        logger.exception(f"Pipeline crashed for City: {city}. Error: {e}")
        return False

def main():
    """Main CLI parser orchestrator."""
    parser = argparse.ArgumentParser(
        description="OSM Ingestion & Feature Engineering CLI Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--city",
        type=str,
        choices=Config.CITIES,
        help="Target city for processing (Bengaluru, Hyderabad, Pune)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sequentially process all cities (Bengaluru, Hyderabad, Pune)"
    )
    
    parser.add_argument(
        "--verify-setup",
        action="store_true",
        help="Verifies dependencies and check boundary file structures"
    )
    
    parser.add_argument(
        "--grid-size",
        type=int,
        help="Override grid cell size resolution in meters (e.g. 1000)"
    )
    
    args = parser.parse_args()
    
    if args.verify_setup:
        setup_ok = verify_setup()
        sys.exit(0 if setup_ok else 1)
        
    if not (args.city or args.all):
        parser.print_help()
        sys.exit(0)
        
    # Route execution
    if args.all:
        logger.info("Executing batch processing mode (--all).")
        success_count = 0
        for city in Config.CITIES:
            if run_osm_pipeline(city, grid_size_override=args.grid_size):
                success_count += 1
        logger.info(f"Batch processing completed. Successful pipelines: {success_count}/{len(Config.CITIES)}")
        sys.exit(0 if success_count == len(Config.CITIES) else 1)
    else:
        success = run_osm_pipeline(args.city, grid_size_override=args.grid_size)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
