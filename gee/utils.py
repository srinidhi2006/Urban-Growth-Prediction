"""
Spatial and Data Utilities for Google Earth Engine.
Handles loading administrative boundary geometries from GeoJSON or GAUL collections.
"""

import json
from pathlib import Path
import ee
from loguru import logger
from config import Config

# Standard mapping from city query name to GAUL Level 2 administrative district names
GAUL_CITY_MAPPING = {
    "Bengaluru": "Bangalore Urban",
    "Hyderabad": "Hyderabad",
    "Pune": "Pune"
}

def load_geojson_geometry(file_path: Path) -> ee.Geometry:
    """Parses a local GeoJSON file and converts its geometries to an Earth Engine geometry.
    
    Args:
        file_path (Path): Path to the GeoJSON file.
        
    Returns:
        ee.Geometry: The parsed GEE Geometry.
        
    Raises:
        ValueError: If the GeoJSON layout is empty or unparsable.
    """
    logger.info(f"Loading local boundary coordinates from GeoJSON: {file_path.name}")
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            
        # Inspect GeoJSON format and extract geometry dict
        if data.get("type") == "FeatureCollection":
            features = data.get("features", [])
            if not features:
                raise ValueError("GeoJSON FeatureCollection contains no features.")
            geom_dict = features[0].get("geometry")
        elif data.get("type") == "Feature":
            geom_dict = data.get("geometry")
        else:
            # Assume raw geometry coordinate dictionary
            geom_dict = data
            
        if not geom_dict:
            raise ValueError("No valid geometry found inside the GeoJSON structure.")
            
        ee_geometry = ee.Geometry(geom_dict)
        return ee_geometry
        
    except Exception as e:
        logger.error(f"Failed to parse GeoJSON boundary at {file_path}: {e}")
        raise ValueError(f"Invalid GeoJSON shape structure: {e}") from e

def get_city_boundary(city_name: str) -> ee.Geometry:
    """Retrieves the administrative boundary geometry for a target city.
    
    Tries to locate a local GeoJSON file first. If unavailable, falls back to querying 
    the GEE GAUL (Global Administrative Unit Layers) dataset. If both fail, raises an error.
    
    Args:
        city_name (str): Name of the city (e.g. 'Bengaluru', 'Hyderabad', 'Pune').
        
    Returns:
        ee.Geometry: The administrative boundary boundary.
        
    Raises:
        ValueError: If no valid boundary shape is retrieved.
    """
    normalized_name = city_name.strip()
    
    # 1. Attempt Local GeoJSON Load
    # Check for both capitalized and lower-cased filename options
    filename_variants = [f"{normalized_name}.geojson", f"{normalized_name.lower()}.geojson"]
    for filename in filename_variants:
        geojson_path = Config.BOUNDARIES_DIR / filename
        if geojson_path.exists():
            try:
                geom = load_geojson_geometry(geojson_path)
                logger.success(f"Successfully loaded local boundary geometry for {normalized_name}.")
                return geom
            except ValueError:
                logger.warning(f"Local file {filename} was unparsable. Trying fallbacks...")
                
    # 2. Fallback to GAUL Level 2 Ingestion
    gaul_name = GAUL_CITY_MAPPING.get(normalized_name)
    if gaul_name:
        try:
            logger.info(f"Local GeoJSON shape not found. Querying GAUL Level 2 for ADM2: '{gaul_name}'")
            gaul_collection = ee.FeatureCollection("FAO/GAUL/2015/level2")
            filtered = gaul_collection.filter(ee.Filter.eq("ADM2_NAME", gaul_name))
            
            # Count elements in collection
            size = filtered.size().getInfo()
            if size > 0:
                geom = filtered.geometry()
                logger.success(f"Successfully queried boundary geometry for {normalized_name} via GAUL.")
                return geom
            else:
                logger.warning(f"No GAUL district found matching ADM2_NAME = '{gaul_name}'.")
        except Exception as e:
            logger.error(f"Earth Engine query for GAUL collection failed: {e}")
            
    # 3. Fail-fast Termination
    err_msg = f"No valid boundary geometry found for city: {normalized_name}"
    logger.critical(err_msg)
    raise ValueError(err_msg)
