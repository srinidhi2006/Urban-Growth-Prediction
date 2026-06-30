"""
Sentinel and OSM Feature Extractor Module.
Calculates zonal statistics (mean NDVI, mean NDBI, mean NDWI) for OSM grid cells.
Merges extracted raster features into the existing OSM features.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from loguru import logger
from config import Config

def extract_raster_features(city: str, year: int) -> bool:
    """Extracts mean NDVI, NDBI, and NDWI raster values for each grid cell polygon,
    and merges them with existing OSM features.
    
    Args:
        city (str): Target city name.
        year (int): Year of the Sentinel composite.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    logger.info(f"[{city}] Starting raster feature extraction for {year}...")
    
    # 1. Paths
    grid_path = Config.INTERIM_DIR / "osm" / city / "grid.geojson"
    osm_path = Config.FEATURES_DIR / f"osm_features_{city}.csv"
    
    processed_dir = Config.PROCESSED_DIR / "sentinel" / city / str(year)
    ndvi_path = processed_dir / f"{city}_{year}_NDVI.tif"
    ndbi_path = processed_dir / f"{city}_{year}_NDBI.tif"
    ndwi_path = processed_dir / f"{city}_{year}_NDWI.tif"
    
    output_path = Config.FEATURES_DIR / f"ml_features_{city}_{year}.csv"
    
    # Verify inputs
    for path in [grid_path, osm_path, ndvi_path, ndbi_path, ndwi_path]:
        if not path.exists():
            logger.error(f"[{city}] Feature extraction failed: File does not exist at: {path}")
            return False
            
    try:
        # Load grid
        grid_gdf = gpd.read_file(grid_path)
        
        # Load OSM features
        osm_df = pd.read_csv(osm_path)
        
        # We will loop over each cell and compute mean NDVI, NDBI, NDWI
        mean_ndvis = []
        mean_ndbis = []
        mean_ndwis = []
        
        # Open rasters
        with rasterio.open(ndvi_path) as ndvi_src, \
             rasterio.open(ndbi_path) as ndbi_src, \
             rasterio.open(ndwi_path) as ndwi_src:
             
            # Standardize projection if necessary
            if grid_gdf.crs != ndvi_src.crs:
                logger.info(f"[{city}] Reprojecting grid CRS to match NDVI raster CRS...")
                grid_gdf = grid_gdf.to_crs(ndvi_src.crs)
                
            total_cells = len(grid_gdf)
            
            for idx, row in grid_gdf.iterrows():
                geom = row.geometry
                
                # Check for None geometries
                if geom is None or geom.is_empty:
                    mean_ndvis.append(np.nan)
                    mean_ndbis.append(np.nan)
                    mean_ndwis.append(np.nan)
                    continue
                
                # Clip NDVI
                try:
                    ndvi_arr, _ = mask(ndvi_src, [geom], crop=True)
                    ndvi_arr = ndvi_arr[0]
                    valid_ndvi = ndvi_arr[~np.isnan(ndvi_arr) & (ndvi_arr != ndvi_src.nodata)]
                    mean_ndvi = float(np.mean(valid_ndvi)) if len(valid_ndvi) > 0 else np.nan
                except Exception:
                    mean_ndvi = np.nan
                
                # Clip NDBI
                try:
                    ndbi_arr, _ = mask(ndbi_src, [geom], crop=True)
                    ndbi_arr = ndbi_arr[0]
                    valid_ndbi = ndbi_arr[~np.isnan(ndbi_arr) & (ndbi_arr != ndbi_src.nodata)]
                    mean_ndbi = float(np.mean(valid_ndbi)) if len(valid_ndbi) > 0 else np.nan
                except Exception:
                    mean_ndbi = np.nan
                    
                # Clip NDWI
                try:
                    ndwi_arr, _ = mask(ndwi_src, [geom], crop=True)
                    ndwi_arr = ndwi_arr[0]
                    valid_ndwi = ndwi_arr[~np.isnan(ndwi_arr) & (ndwi_arr != ndwi_src.nodata)]
                    mean_ndwi = float(np.mean(valid_ndwi)) if len(valid_ndwi) > 0 else np.nan
                except Exception:
                    mean_ndwi = np.nan
                    
                mean_ndvis.append(mean_ndvi)
                mean_ndbis.append(mean_ndbi)
                mean_ndwis.append(mean_ndwi)
                
        # Build features DataFrame
        extracted_df = pd.DataFrame({
            "grid_id": grid_gdf["grid_id"],
            "mean_ndvi": mean_ndvis,
            "mean_ndbi": mean_ndbis,
            "mean_ndwi": mean_ndwis
        })
        
        # Merge with existing OSM features on grid_id
        # Make sure grid_id data types match
        extracted_df["grid_id"] = extracted_df["grid_id"].astype(str)
        osm_df["grid_id"] = osm_df["grid_id"].astype(str)
        
        merged_df = pd.merge(osm_df, extracted_df, on="grid_id", how="left")
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        logger.success(f"[{city}] Successfully saved ML features table to: {output_path.name}")
        
        return True
        
    except Exception as e:
        logger.exception(f"[{city}] Feature extraction crashed: {e}")
        return False
