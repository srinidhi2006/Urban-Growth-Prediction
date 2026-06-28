"""
OSM Feature Validator Stage.
Runs strict assertions and data integrity tests on calculated ML features,
generating a detailed validation report JSON.
"""

import json
from datetime import datetime
import pandas as pd
import geopandas as gpd
from loguru import logger
from config import Config

def validate_osm_features(city: str) -> bool:
    """Performs validation assertions on the engineered OSM feature tables for a city.
    
    Args:
        city (str): Target city name.
        
    Returns:
        bool: True if validation assertions pass, False if checks report critical errors.
    """
    logger.info(f"Running validation checks on engineered OSM features for {city}...")
    
    features_dir = Config.FEATURES_DIR
    csv_path = features_dir / f"osm_features_{city}.csv"
    geojson_path = features_dir / f"osm_features_{city}.geojson"
    report_path = features_dir / f"validation_report_{city}.json"
    
    # 1. Assert file availability
    if not csv_path.exists() or not geojson_path.exists():
        logger.error(f"[{city}] Validation failed: CSV or GeoJSON feature outputs are missing.")
        return False
        
    try:
        # Load tables
        df = pd.read_csv(csv_path)
        gdf = gpd.read_file(geojson_path)
        
        passed = True
        checks = {
            "total_cells_validated": len(df),
            "duplicate_geometries_found": 0,
            "invalid_geometries_found": 0,
            "empty_geometries_found": 0,
            "invalid_crs_found": False,
            "zero_area_polygons_found": 0,
            "duplicate_grid_ids_found": 0,
            "negative_values_found": 0,
            "missing_values_found": 0,
            "ratio_bounds_violations": 0
        }
        
        # 2. Check CRS (Should be stored in WGS84 EPSG:4326 for GIS map engines)
        if gdf.crs is None or str(gdf.crs).upper() != "EPSG:4326":
            logger.warning(f"[{city}] Validation Warn: GeoJSON CRS is '{gdf.crs}', expected EPSG:4326.")
            checks["invalid_crs_found"] = True
            passed = False
            
        # 3. Check Geometry issues
        empty_geoms = gdf[gdf.geometry.is_empty]
        checks["empty_geometries_found"] = len(empty_geoms)
        
        invalid_geoms = gdf[~gdf.geometry.is_valid]
        checks["invalid_geometries_found"] = len(invalid_geoms)
        
        # Check duplicate geometries
        dup_geoms = gdf[gdf.geometry.duplicated()]
        checks["duplicate_geometries_found"] = len(dup_geoms)
        
        # Check zero-area cells (project first to meters to check area)
        gdf_projected = gdf.to_crs(Config.PROJECTED_CRS)
        zero_area = gdf_projected[gdf_projected.geometry.area <= 0.0]
        checks["zero_area_polygons_found"] = len(zero_area)
        
        # 4. Check IDs & Missing values
        dup_ids = df[df["grid_id"].duplicated()]
        checks["duplicate_grid_ids_found"] = len(dup_ids)
        
        missing_vals = df.isnull().sum().sum()
        checks["missing_values_found"] = int(missing_vals)
        
        # 5. Check Feature Value boundaries
        # Checks for negative values
        numeric_cols = [
            "building_count", "building_density", "building_area_ratio", 
            "road_length", "road_density", "road_intersection_count", 
            "intersection_density", "distance_to_highway", "green_area", 
            "green_ratio", "distance_to_center"
        ]
        
        negatives = 0
        for col in numeric_cols:
            if col in df.columns:
                negatives += (df[col] < 0.0).sum()
        checks["negative_values_found"] = int(negatives)
        
        # Checks for ratio values (should reside inside [0.0, 1.0])
        ratio_violations = 0
        for col in ["building_area_ratio", "green_ratio"]:
            if col in df.columns:
                # Add tolerance for rounding/epsilon errors
                ratio_violations += ((df[col] < -1e-6) | (df[col] > 1.000001)).sum()
        checks["ratio_bounds_violations"] = int(ratio_violations)
        
        # Evaluate final status
        if (checks["duplicate_geometries_found"] > 0 or 
            checks["invalid_geometries_found"] > 0 or 
            checks["empty_geometries_found"] > 0 or 
            checks["zero_area_polygons_found"] > 0 or 
            checks["duplicate_grid_ids_found"] > 0 or 
            checks["negative_values_found"] > 0 or 
            checks["missing_values_found"] > 0 or 
            checks["ratio_bounds_violations"] > 0):
            passed = False
            
        report = {
            "city": city,
            "validation_timestamp": datetime.utcnow().isoformat() + "Z",
            "passed": passed,
            "checks": checks
        }
        
        # Write validation report to disk
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        if passed:
            logger.success(f"[{city}] All validation checks PASSED successfully.")
        else:
            logger.error(
                f"[{city}] Validation checks FAILED! Violations logged: "
                f"Negative counts: {checks['negative_values_found']}, "
                f"Missing fields: {checks['missing_values_found']}, "
                f"Ratio limit errors: {checks['ratio_bounds_violations']}, "
                f"Invalid geoms: {checks['invalid_geometries_found']}."
            )
            
        return passed
        
    except Exception as e:
        logger.exception(f"[{city}] Validator stage crashed: {e}")
        return False
