"""
Sentinel Tiling Module.
Generates square tiles over administrative boundaries for tiled satellite ingestion.
Saves main GeoJSON layer, individual tile boundary GeoJSONs, and metadata summaries.
"""

import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from loguru import logger
from config import Config

def generate_sentinel_tiles(city: str, boundary_gdf: gpd.GeoDataFrame, tile_size_meters: int = None) -> bool:
    """Generates square grid tiles over a city boundary, assigning metadata and outputting files.
    
    Args:
        city (str): Target city name.
        boundary_gdf (gpd.GeoDataFrame): City boundary in EPSG:4326.
        tile_size_meters (int, optional): Grid step size in meters. Defaults to Config.SENTINEL_TILE_SIZE_METERS.
        
    Returns:
        bool: True if tile generation completed successfully, False otherwise.
    """
    tile_size = tile_size_meters if tile_size_meters is not None else Config.SENTINEL_TILE_SIZE_METERS
    logger.info(f"[{city}] Starting Sentinel tile generation with size {tile_size}m...")
    
    try:
        # 1. Project boundary to projected CRS (EPSG:3857)
        if boundary_gdf.crs != Config.PROJECTED_CRS:
            boundary_projected = boundary_gdf.to_crs(Config.PROJECTED_CRS)
        else:
            boundary_projected = boundary_gdf
            
        boundary_geom = boundary_projected.geometry.iloc[0]
        minx, miny, maxx, maxy = boundary_geom.bounds
        
        # 2. Generate regular grid coordinates
        x_coords = np.arange(minx, maxx, tile_size)
        y_coords = np.arange(miny, maxy, tile_size)
        
        cells = []
        for x in x_coords:
            for y in y_coords:
                cells.append(box(x, y, x + tile_size, y + tile_size))
                
        grid_gdf = gpd.GeoDataFrame(geometry=cells, crs=Config.PROJECTED_CRS)
        
        # 3. Keep only tiles intersecting the administrative boundary
        intersecting_mask = grid_gdf.geometry.intersects(boundary_geom)
        tiles_gdf = grid_gdf[intersecting_mask].copy()
        
        if tiles_gdf.empty:
            logger.error(f"[{city}] No tiles intersected the boundary. Check coordinates.")
            return False
            
        # 4. Format metadata
        tiles_gdf = tiles_gdf.reset_index(drop=True)
        tiles_gdf["tile_id"] = tiles_gdf.index
        tiles_gdf["city"] = city
        tiles_gdf["tile_name"] = tiles_gdf["tile_id"].apply(lambda tid: f"{city}_tile_{tid:03d}")
        tiles_gdf["status"] = "pending"
        
        # Reorder columns to place geometry last
        tiles_gdf = tiles_gdf[["tile_id", "city", "tile_name", "status", "geometry"]]
        
        # 5. Create output directories
        interim_sentinel_dir = Config.INTERIM_DIR / "sentinel" / city
        boundaries_dir = interim_sentinel_dir / "tile_boundaries"
        interim_sentinel_dir.mkdir(parents=True, exist_ok=True)
        boundaries_dir.mkdir(parents=True, exist_ok=True)
        
        # 6. Save Main GeoJSON Layer
        main_layer_path = interim_sentinel_dir / "tiles.geojson"
        tiles_gdf.to_file(main_layer_path, driver="GeoJSON")
        logger.info(f"[{city}] Saved main tiles layer ({len(tiles_gdf)} tiles) to {main_layer_path}")
        
        # 7. Save Individual Tile GeoJSON Files
        # First clean out existing files in the directory to avoid mix-ups
        for existing_file in boundaries_dir.glob("tile_*.geojson"):
            existing_file.unlink()
            
        for idx, row in tiles_gdf.iterrows():
            tile_row = gpd.GeoDataFrame([row], crs=Config.PROJECTED_CRS)
            tile_path = boundaries_dir / f"tile_{idx:03d}.geojson"
            tile_row.to_file(tile_path, driver="GeoJSON")
            
        logger.info(f"[{city}] Saved {len(tiles_gdf)} individual tile boundary GeoJSON files to {boundaries_dir}")
        
        # 8. Calculate boundary area in sqkm using local UTM Zone projection to avoid Web Mercator distortion
        # Get WGS84 coordinates of the centroid to compute local UTM EPSG code
        centroid_wgs84 = boundary_projected.geometry.centroid.to_crs("EPSG:4326").iloc[0]
        lon, lat = centroid_wgs84.x, centroid_wgs84.y
        utm_zone = int((lon + 180) / 6) + 1
        is_northern = lat >= 0
        epsg_code = f"EPSG:326{utm_zone:02d}" if is_northern else f"EPSG:327{utm_zone:02d}"
        
        boundary_utm = boundary_gdf.to_crs(epsg_code)
        boundary_area_sqkm = float(boundary_utm.geometry.iloc[0].area / 1_000_000.0)
        
        # 9. Save Summary JSON
        summary_path = interim_sentinel_dir / "tile_summary.json"
        summary_data = {
            "city": city,
            "tile_size_m": int(tile_size),
            "tiles_generated": int(len(tiles_gdf)),
            "crs": Config.PROJECTED_CRS,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "boundary_area_sqkm": round(boundary_area_sqkm, 2)
        }
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)
            
        logger.success(f"[{city}] Sentinel tile generation pipeline finished successfully.")
        return True
        
    except Exception as e:
        logger.exception(f"[{city}] Tile Generator stage crashed: {e}")
        return False
