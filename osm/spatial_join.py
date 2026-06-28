"""
OSM Spatial Join Stage.
Spatially joins clean geometries to the generated grid, performing geometric overlays
and clipping shapes to cell borders to prevent area/length double-counting across borders.
"""

from pathlib import Path
import geopandas as gpd
from loguru import logger
from config import Config

def join_spatial_data(city: str) -> bool:
    """Clips cleaned layers to the grid cell borders, mapping geometries to grid_ids.
    
    Args:
        city (str): Target city name.
        
    Returns:
        bool: True if spatial join overlays completed successfully, False otherwise.
    """
    logger.info(f"Starting metric spatial join and boundary overlay clipping for {city}...")
    
    interim_city_dir = Config.INTERIM_DIR / "osm" / city
    grid_path = interim_city_dir / "grid.geojson"
    
    # Verify grid exists
    if not grid_path.exists():
        logger.error(f"[{city}] Spatial grid not found at: {grid_path}")
        return False
        
    grid = gpd.read_file(grid_path)
    
    layers = {
        "buildings": interim_city_dir / "buildings_cleaned.geojson",
        "green_areas": interim_city_dir / "green_areas_cleaned.geojson",
        "roads": interim_city_dir / "roads_cleaned.geojson"
    }
    
    try:
        for layer_name, file_path in layers.items():
            if not file_path.exists():
                logger.error(f"[{city}] Cleaned layer file not found: {file_path}")
                return False
                
            gdf = gpd.read_file(file_path)
            
            logger.info(f"[{city}] Executing overlay intersection for {layer_name}...")
            
            if gdf.empty:
                logger.warning(f"[{city}] Cleaned {layer_name} is empty. Creating empty joined representation.")
                # Return empty GDF but with grid_id column
                gdf_joined = gpd.GeoDataFrame(columns=["grid_id", "geometry"], crs=Config.PROJECTED_CRS)
            else:
                # Use overlay intersection to clip geometries precisely to cell borders
                # This ensures that area/length counts are mathematically correct within each cell.
                gdf_joined = gpd.overlay(gdf, grid, how="intersection")
                
            out_path = interim_city_dir / f"{layer_name}_joined.geojson"
            gdf_joined.to_file(out_path, driver="GeoJSON")
            logger.info(f"[{city}] Saved joined {layer_name} ({len(gdf_joined)} segments) to {out_path.name}")
            
        logger.success(f"[{city}] Spatial join overlays completed successfully.")
        return True
        
    except Exception as e:
        logger.exception(f"[{city}] Spatial Join stage crashed: {e}")
        return False
