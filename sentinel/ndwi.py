"""
Sentinel NDWI Generation Module.
Computes Normalized Difference Water Index (NDWI) from processed Sentinel-2 mosaics.
Supports safe division and visual maps generation.
"""

import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loguru import logger
from config import Config

def generate_ndwi(city: str, year: int) -> bool:
    """Generates a single-band NDWI float32 GeoTIFF from the Sentinel-2 mosaic.
    
    NDWI = (B3 - B8) / (B3 + B8) where Band 2 is Green (B3) and Band 4 is NIR (B8).
    
    Args:
        city (str): Target city name.
        year (int): Year of the mosaic.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    logger.info(f"[{city}] Starting NDWI generation for {year}...")
    
    processed_dir = Config.PROCESSED_DIR / "sentinel" / city / str(year)
    mosaic_path = processed_dir / f"{city}_{year}_mosaic.tif"
    
    if not mosaic_path.exists():
        logger.error(f"[{city}] Source mosaic TIFF not found at: {mosaic_path}")
        return False
        
    try:
        # 1. Load Mosaic
        logger.info(f"[{city}] Loaded mosaic: {mosaic_path.name}")
        with rasterio.open(mosaic_path) as src:
            logger.info(f"[{city}] Reading Green band (Band 2)")
            green = src.read(2).astype(np.float32)
            
            logger.info(f"[{city}] Reading NIR band (Band 4)")
            nir = src.read(4).astype(np.float32)
            
            out_meta = src.meta.copy()
            crs_val = src.crs
            transform_val = src.transform
            
        # 2. Compute NDWI
        logger.info(f"[{city}] Computing NDWI")
        denominator = green + nir
        
        # Apply safe division avoiding divide-by-zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = np.where(denominator != 0.0, (green - nir) / denominator, np.nan)
            
        # Apply no-data masking where source has 0 reflectance (e.g. background mask pixels)
        ndwi[(green == 0.0) & (nir == 0.0)] = np.nan
        
        # 3. Save GeoTIFF
        logger.info(f"[{city}] Saving GeoTIFF")
        ndwi_path = processed_dir / f"{city}_{year}_NDWI.tif"
        
        out_meta.update({
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "height": ndwi.shape[0],
            "width": ndwi.shape[1],
            "transform": transform_val,
            "crs": crs_val,
            "nodata": np.nan
        })
        
        with rasterio.open(ndwi_path, "w", **out_meta) as dest:
            dest.write(ndwi.astype(np.float32), 1)
            
        logger.success(f"[{city}] NDWI GeoTIFF saved at: {ndwi_path.name}")
        
        # 4. Generate Preview Image
        logger.info(f"[{city}] Generating preview")
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        im = ax.imshow(ndwi, cmap='Blues', vmin=-0.5, vmax=0.5)
        ax.set_title(f"{city} NDWI ({year})", fontsize=14, weight='bold', pad=15)
        ax.axis('off')
        
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
        cbar.set_label("NDWI Value", fontsize=10, labelpad=10)
        cbar.ax.tick_params(labelsize=8)
        
        preview_path = processed_dir / f"{city}_{year}_NDWI_preview.png"
        plt.savefig(preview_path, bbox_inches='tight', dpi=150)
        plt.close()
        logger.success(f"[{city}] NDWI Preview saved at: {preview_path.name}")
        
        return True
        
    except Exception as e:
        logger.exception(f"[{city}] Failed to generate NDWI: {e}")
        return False
