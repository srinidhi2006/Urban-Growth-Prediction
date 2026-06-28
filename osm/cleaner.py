"""
OSM Cleaner Stage.
Loads raw downloads, repairs invalid geometries using make_valid, projects coordinates
to EPSG:3857, clips them to the administrative study area boundary, and writes clean layers.
"""

from pathlib import Path
import geopandas as gpd
from shapely.validation import make_valid
from loguru import logger
from config import Config

def clean_osm_data(city: str, boundary_gdf: gpd.GeoDataFrame) -> bool:
    """Cleans, repairs, projects, and clips raw OSM vector datasets for a city.
    
    Args:
        city (str): Target city name.
        boundary_gdf (gpd.GeoDataFrame): Administrative boundary (EPSG:4326).
        
    Returns:
        bool: True if cleaning completed successfully, False otherwise.
    """
    logger.info(f"Starting geometry cleaning and CRS projection for {city}...")
    
    raw_city_dir = Config.RAW_OSM_DIR / city
    interim_city_dir = Config.INTERIM_DIR / "osm" / city
    interim_city_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    raw_files = {
        "buildings": raw_city_dir / "buildings.geojson",
        "green_areas": raw_city_dir / "green_areas.geojson",
        "roads": raw_city_dir / "roads.geojson"
    }
    
    # Project boundary to EPSG:3857 for metric-based calculations and clipping
    boundary_projected = boundary_gdf.to_crs(Config.PROJECTED_CRS)
    boundary_geom = boundary_projected.geometry.iloc[0]
    
    try:
        for layer_name, file_path in raw_files.items():
            if not file_path.exists():
                logger.error(f"[{city}] Raw {layer_name} file not found at: {file_path}")
                return False
                
            logger.info(f"[{city}] Cleaning and projecting {layer_name}...")
            gdf = gpd.read_file(file_path)
            
            if gdf.empty:
                logger.warning(f"[{city}] Raw {layer_name} is empty. Creating empty cleaned dataset.")
                gdf_cleaned = gpd.GeoDataFrame(geometry=[], crs=Config.PROJECTED_CRS)
            else:
                # 1. Geometry repair
                gdf["geometry"] = gdf["geometry"].apply(
                    lambda geom: geom if geom is not None and geom.is_valid else (make_valid(geom) if geom is not None else None)
                )
                
                # 2. Filter out null/empty geometries
                gdf = gdf[gdf.geometry.notnull() & (~gdf.geometry.is_empty)]
                
                # Filter by geometry type to prevent mixed geometry overlay errors
                if layer_name in ["buildings", "green_areas"]:
                    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
                elif layer_name == "roads":
                    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]
                
                # 3. Project to target projected CRS
                gdf = gdf.to_crs(Config.PROJECTED_CRS)
                
                # 4. Clip to administrative boundaries
                gdf_cleaned = gpd.clip(gdf, boundary_projected)
                
            # Save cleaned dataset
            out_path = interim_city_dir / f"{layer_name}_cleaned.geojson"
            gdf_cleaned.to_file(out_path, driver="GeoJSON")
            logger.info(f"[{city}] Saved cleaned {layer_name} ({len(gdf_cleaned)} features) to {out_path.name}")
            
        logger.success(f"[{city}] Completed geometry cleaning stage.")
        return True
        
    except Exception as e:
        logger.exception(f"[{city}] Cleaner stage crashed: {e}")
        return False
