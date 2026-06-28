"""
OSM Grid Generator Stage.
Generates a regular metric tessellation grid of cell size grid_size_meters over the study boundary area,
filtering grid cells that intersect the boundary and indexing them with grid_id.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import box
from loguru import logger
from config import Config

def generate_spatial_grid(city: str, boundary_gdf: gpd.GeoDataFrame, grid_size_meters: int = None) -> bool:
    """Generates a regular projected tessellated grid of grid_size_meters length over a city boundary.
    
    Args:
        city (str): Name of the target city.
        boundary_gdf (gpd.GeoDataFrame): Administrative boundary (EPSG:4326).
        grid_size_meters (int, optional): Grid cell width override. Defaults to Config.OSM_GRID_SIZE_METERS.
        
    Returns:
        bool: True if grid generation completed successfully, False otherwise.
    """
    grid_size = grid_size_meters if grid_size_meters else Config.OSM_GRID_SIZE_METERS
    logger.info(f"Starting spatial grid generation for {city} (cell size = {grid_size}m)...")
    
    interim_city_dir = Config.INTERIM_DIR / "osm" / city
    interim_city_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Project boundary to projected CRS
        boundary_projected = boundary_gdf.to_crs(Config.PROJECTED_CRS)
        boundary_geom = boundary_projected.geometry.iloc[0]
        
        # Get bounding box of the boundary
        minx, miny, maxx, maxy = boundary_geom.bounds
        
        # Create regular interval points
        x_coords = np.arange(minx, maxx, grid_size)
        y_coords = np.arange(miny, maxy, grid_size)
        
        cells = []
        grid_id = 0
        
        for x in x_coords:
            for y in y_coords:
                # Create a square bounding box cell
                cell_box = box(x, y, x + grid_size, y + grid_size)
                
                # Verify that the cell intersects with the administrative boundary geometry
                if cell_box.intersects(boundary_geom):
                    cells.append({
                        "grid_id": grid_id,
                        "geometry": cell_box
                    })
                    grid_id += 1
                    
        if not cells:
            logger.error(f"[{city}] No grid cells generated inside boundary bounding envelope.")
            return False
            
        # Convert list of dicts to GeoDataFrame
        grid_gdf = gpd.GeoDataFrame(cells, crs=Config.PROJECTED_CRS)
        
        # Save output grid
        grid_path = interim_city_dir / "grid.geojson"
        grid_gdf.to_file(grid_path, driver="GeoJSON")
        
        logger.success(f"[{city}] Successfully generated {len(grid_gdf)} grid cells. Saved to {grid_path.name}")
        return True
        
    except Exception as e:
        logger.exception(f"[{city}] Grid Generator stage crashed: {e}")
        return False
