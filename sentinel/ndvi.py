"""
Sentinel NDVI Generation Module.
Computes scientifically correct NDVI rasters from processed Sentinel-2 mosaics.
Supports safe division, visual maps generation, and metadata compilation.
"""

import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to prevent GUI thread issues
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from loguru import logger
from config import Config
from gee.export import get_git_commit, PIPELINE_VERSION

def generate_ndvi(city: str, year: int) -> bool:
    """Generates a single-band NDVI float32 GeoTIFF from the Sentinel-2 mosaic.
    
    NDVI = (B8 - B4) / (B8 + B4) where Band 3 is Red (B4) and Band 4 is NIR (B8).
    
    Args:
        city (str): Target city.
        year (int): Year of the mosaic.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    logger.info(f"[{city}] Starting NDVI generation for {year}...")
    start_time = time.time()
    
    processed_dir = Config.PROCESSED_DIR / "sentinel" / city / str(year)
    mosaic_path = processed_dir / f"{city}_{year}_mosaic.tif"
    
    if not mosaic_path.exists():
        logger.error(f"[{city}] Source mosaic TIFF not found at: {mosaic_path}")
        return False
        
    try:
        # 1. Load Mosaic
        logger.info(f"[{city}] Loaded mosaic: {mosaic_path.name}")
        with rasterio.open(mosaic_path) as src:
            logger.info(f"[{city}] Reading Red band (Band 3)")
            red = src.read(3).astype(np.float32)
            
            logger.info(f"[{city}] Reading NIR band (Band 4)")
            nir = src.read(4).astype(np.float32)
            
            out_meta = src.meta.copy()
            crs_val = src.crs
            is_geo = src.crs.is_geographic
            transform_val = src.transform
            
        # 2. Compute NDVI
        logger.info(f"[{city}] Computing NDVI")
        denominator = nir + red
        
        # Apply safe division avoiding divide-by-zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = np.where(denominator != 0.0, (nir - red) / denominator, np.nan)
            
        # Apply no-data masking where source has 0 reflectance (e.g. background mask pixels)
        ndvi[(red == 0.0) & (nir == 0.0)] = np.nan
        
        # 3. Save GeoTIFF
        logger.info(f"[{city}] Saving GeoTIFF")
        ndvi_path = processed_dir / f"{city}_{year}_NDVI.tif"
        
        out_meta.update({
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "height": ndvi.shape[0],
            "width": ndvi.shape[1],
            "transform": transform_val,
            "crs": crs_val,
            "nodata": np.nan
        })
        
        with rasterio.open(ndvi_path, "w", **out_meta) as dest:
            dest.write(ndvi.astype(np.float32), 1)
            
        logger.success(f"[{city}] NDVI GeoTIFF saved at: {ndvi_path.name}")
        
        # 4. Generate Preview Image
        logger.info(f"[{city}] Generating preview")
        # Custom colormap: Brown -> Light Brown -> Light Yellow -> Light Green -> Dark Green
        colors = ["#8B4513", "#D2B48C", "#FFFFE0", "#90EE90", "#006400"]
        ndvi_cmap = LinearSegmentedColormap.from_list("ndvi_custom", colors, N=100)
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        # Use standard NDVI range min/max bounds for urban contrast
        im = ax.imshow(ndvi, cmap=ndvi_cmap, vmin=-0.1, vmax=0.7)
        ax.set_title(f"{city} NDVI ({year})", fontsize=14, weight='bold', pad=15)
        ax.axis('off')
        
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
        cbar.set_label("NDVI Value", fontsize=10, labelpad=10)
        cbar.ax.tick_params(labelsize=8)
        
        preview_path = processed_dir / f"{city}_{year}_NDVI_preview.png"
        plt.savefig(preview_path, bbox_inches='tight', dpi=150)
        plt.close()
        logger.success(f"[{city}] NDVI Preview saved at: {preview_path.name}")
        
        # 5. Extract statistics and save metadata JSON
        valid_mask = ~np.isnan(ndvi)
        valid_data = ndvi[valid_mask]
        
        min_val = float(np.min(valid_data)) if len(valid_data) > 0 else 0.0
        max_val = float(np.max(valid_data)) if len(valid_data) > 0 else 0.0
        mean_val = float(np.mean(valid_data)) if len(valid_data) > 0 else 0.0
        std_val = float(np.std(valid_data)) if len(valid_data) > 0 else 0.0
        
        valid_pixels = int(np.sum(valid_mask))
        nodata_pixels = int(np.sum(~valid_mask))
        processing_duration = time.time() - start_time
        
        meta_data = {
            "city": city,
            "year": year,
            "formula": "(B8-B4)/(B8+B4)",
            "band_red": "B4",
            "band_nir": "B8",
            "minimum_ndvi": round(min_val, 4),
            "maximum_ndvi": round(max_val, 4),
            "mean_ndvi": round(mean_val, 4),
            "std_ndvi": round(std_val, 4),
            "valid_pixels": valid_pixels,
            "nodata_pixels": nodata_pixels,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "pipeline_version": PIPELINE_VERSION,
            "git_commit": get_git_commit()
        }
        
        metadata_path = processed_dir / "ndvi_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as mf:
            json.dump(meta_data, mf, indent=4)
            
        logger.success(f"[{city}] NDVI Metadata statistics saved at: {metadata_path.name}")
        return True
        
    except Exception as e:
        logger.exception(f"[{city}] Failed to generate NDVI: {e}")
        return False
