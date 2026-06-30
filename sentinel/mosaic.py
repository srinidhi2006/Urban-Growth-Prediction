"""
Sentinel Mosaic Generation Module.
Merges downloaded raw Sentinel-2 tiles into a seamless city composite GeoTIFF.
Verifies raster resolution, bands count, datatypes, and CRS consistency before merging.
Saves the processed mosaic, preview image, and metadata JSON.
"""

import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import rasterio
from rasterio.merge import merge
from PIL import Image
from loguru import logger
from config import Config
from gee.export import get_git_commit, PIPELINE_VERSION

def create_mosaic(city: str, year: int) -> bool:
    """Merges all downloaded tiles for a city and year into a single GeoTIFF.
    
    Verifies input rasters are consistent (CRS, resolution, band count, and dtype) before merging.
    
    Args:
        city (str): Target city.
        year (int): Ingestion calendar year.
        
    Returns:
        bool: True if mosaic created successfully, False otherwise.
    """
    logger.info(f"[{city}] Starting Sentinel-2 mosaic generation for {year}...")
    start_time = time.time()
    
    tiles_dir = Config.RAW_SENTINEL_DIR / city / str(year) / "tiles"
    if not tiles_dir.exists():
        logger.error(f"[{city}] Tiles directory does not exist at: {tiles_dir}")
        return False
        
    # Find all downloaded tile TIFF files
    tile_paths = sorted(list(tiles_dir.glob("tile_*.tif")))
    if not tile_paths:
        logger.error(f"[{city}] No tile TIFF files found in: {tiles_dir}")
        return False
        
    logger.info(f"[{city}] Found {len(tile_paths)} tiles to verify and merge.")
    
    # 1. Verification of raster parameters before merge
    src_files = []
    try:
        reference_crs = None
        reference_res = None
        reference_dtype = None
        
        for fp in tile_paths:
            if not fp.exists():
                logger.error(f"[{city}] Missing expected tile file: {fp}")
                return False
                
            src = rasterio.open(fp)
            src_files.append(src)
            
            # Check bands count
            if src.count != 6:
                logger.error(f"[{city}] Pre-merge verification failed: Raster {fp.name} has {src.count} bands, expected 6.")
                return False
                
            # Initialize references from the first tile
            if reference_crs is None:
                reference_crs = str(src.crs)
                reference_res = src.res
                reference_dtype = src.dtypes[0]
                
            # Check CRS consistency
            if str(src.crs) != reference_crs:
                logger.error(
                    f"[{city}] Pre-merge verification failed: CRS mismatch in raster {fp.name}. "
                    f"Got {src.crs}, expected {reference_crs}."
                )
                return False
                
            # Check resolution consistency
            if src.res != reference_res:
                logger.error(
                    f"[{city}] Pre-merge verification failed: Pixel size resolution mismatch in raster {fp.name}. "
                    f"Got {src.res}, expected {reference_res}."
                )
                return False
                
            # Check data type consistency
            if src.dtypes[0] != reference_dtype:
                logger.error(
                    f"[{city}] Pre-merge verification failed: Data type mismatch in raster {fp.name}. "
                    f"Got {src.dtypes[0]}, expected {reference_dtype}."
                )
                return False
                
        # 2. Merge all rasters
        logger.info(f"[{city}] Pre-merge validation PASSED. Merging rasters...")
        mosaic, out_trans = merge(src_files)
        
        # Read profile and close file descriptors
        out_meta = src_files[0].meta.copy()
        is_geo = src_files[0].crs.is_geographic
        res_val = src_files[0].res[0]
        crs_val = src_files[0].crs
        
        for src in src_files:
            src.close()
        src_files.clear() # clear closed references
        
        # 3. Configure output metadata profile
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": crs_val
        })
        
        # Save merged image
        processed_dir = Config.PROCESSED_DIR / "sentinel" / city / str(year)
        processed_dir.mkdir(parents=True, exist_ok=True)
        mosaic_path = processed_dir / f"{city}_{year}_mosaic.tif"
        
        with rasterio.open(mosaic_path, "w", **out_meta) as dest:
            dest.write(mosaic)
            
        logger.success(f"[{city}] Mosaic saved successfully at: {mosaic_path}")
        
        # 4. Generate stretched RGB preview image
        # Bands: 0=B2(Blue), 1=B3(Green), 2=B4(Red)
        r = mosaic[2] # B4
        g = mosaic[1] # B3
        b = mosaic[0] # B2
        
        # Stretch and scale B4, B3, B2 to 0-255 uint8 range (typical reflectance 0-3000 max)
        rgb = np.dstack([r, g, b])
        rgb = np.clip(rgb / 3000.0, 0, 1)
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        
        preview_path = processed_dir / f"{city}_{year}_preview.png"
        Image.fromarray(rgb_uint8).save(preview_path)
        logger.success(f"[{city}] Stretched RGB preview saved at: {preview_path}")
        
        # 5. Save mosaic metadata file
        metadata_path = processed_dir / "mosaic_metadata.json"
        processing_duration = time.time() - start_time
        file_size_mb = mosaic_path.stat().st_size / (1024 * 1024)
        
        meta_data = {
            "city": city,
            "year": year,
            "tiles_merged": len(tile_paths),
            "bands": 6,
            "width": mosaic.shape[2],
            "height": mosaic.shape[1],
            "crs": str(crs_val),
            "pixel_size": float(res_val),
            "pixel_size_unit": "degrees" if is_geo else "meters",
            "file_size_mb": round(file_size_mb, 2),
            "processing_time_sec": round(processing_duration, 2),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "pipeline_version": PIPELINE_VERSION,
            "git_commit": get_git_commit()
        }
        
        with open(metadata_path, "w", encoding="utf-8") as mf:
            json.dump(meta_data, mf, indent=4)
            
        logger.success(f"[{city}] Mosaic metadata statistics saved at: {metadata_path}")
        return True
        
    except Exception as e:
        logger.exception(f"[{city}] Sentinel Mosaic compilation failed: {e}")
        # Clean up open resources in case of exception
        for src in src_files:
            try:
                src.close()
            except Exception:
                pass
        return False
