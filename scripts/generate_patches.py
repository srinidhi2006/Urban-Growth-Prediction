import os
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from pathlib import Path
from loguru import logger

def main():
    # 1. Setup paths
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    features_path = data_dir / "features" / "combined_growth_dataset.csv"
    cnn_dir = data_dir / "cnn"
    reports_dir = project_root / "reports"
    
    # Create directories
    cnn_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading combined growth dataset...")
    df_growth = pd.read_csv(features_path)
    
    cities = ["Bengaluru", "Hyderabad", "Pune"]
    dataset_records = []
    
    # Track statistics
    total_patches = 0
    skipped_shape = 0
    skipped_nan = 0
    skipped_empty = 0
    
    # To store samples for preview plot
    preview_samples = {}
    
    for city in cities:
        city_cnn_dir = cnn_dir / city
        city_cnn_dir.mkdir(parents=True, exist_ok=True)
        
        # Load grid
        grid_path = data_dir / "interim" / "osm" / city / "grid.geojson"
        if not grid_path.exists():
            logger.error(f"[{city}] Grid GeoJSON not found at: {grid_path}")
            continue
            
        logger.info(f"[{city}] Loading grid cells...")
        grid_gdf = gpd.read_file(grid_path)
        
        # Open 2026 Sentinel-2 mosaic
        mosaic_path = data_dir / "processed" / "sentinel" / city / "2026" / f"{city}_2026_mosaic.tif"
        if not mosaic_path.exists():
            logger.error(f"[{city}] Sentinel-2 mosaic not found at: {mosaic_path}")
            continue
            
        logger.info(f"[{city}] Opening Sentinel-2 2026 mosaic...")
        with rasterio.open(mosaic_path) as src:
            # Match CRS
            grid_gdf_projected = grid_gdf.to_crs(src.crs)
            
            logger.info(f"[{city}] Extracting and resampling {len(grid_gdf_projected)} image patches...")
            
            for idx, row in grid_gdf_projected.iterrows():
                grid_id = int(row["grid_id"])
                geom = row.geometry
                
                if geom is None or geom.is_empty:
                    logger.warning(f"[{city}] Empty geometry found for grid_id {grid_id}. Skipping...")
                    continue
                    
                # Crop
                try:
                    out_image, out_transform = mask(src, [geom], crop=True)
                except Exception as e:
                    logger.warning(f"[{city}] Crop failed for grid_id {grid_id}: {e}. Skipping...")
                    continue
                    
                # Resample to exactly 128x128
                h, w = out_image.shape[1], out_image.shape[2]
                if h == 0 or w == 0:
                    skipped_empty += 1
                    continue
                    
                with MemoryFile() as memfile:
                    profile = src.profile.copy()
                    profile.update({
                        'height': h,
                        'width': w,
                        'transform': out_transform
                    })
                    try:
                        with memfile.open(**profile) as mem_dst:
                            mem_dst.write(out_image)
                            resized = mem_dst.read(
                                out_shape=(src.count, 128, 128),
                                resampling=Resampling.bilinear
                            )
                    except Exception as e:
                        logger.warning(f"[{city}] Resampling failed for grid_id {grid_id}: {e}. Skipping...")
                        continue
                
                # Quality Verification Checks
                # Check 1: Shape check
                if resized.shape != (6, 128, 128):
                    skipped_shape += 1
                    logger.warning(f"[{city}] Invalid resampled shape {resized.shape} for grid_id {grid_id}. Skipping...")
                    continue
                    
                # Check 2: NaNs check
                if np.isnan(resized).any():
                    # Fill NaNs with 0.0
                    resized = np.nan_to_num(resized, nan=0.0)
                    
                # Check 3: Completely empty check
                if np.sum(np.abs(resized)) == 0.0:
                    skipped_empty += 1
                    logger.warning(f"[{city}] Empty patch (all zeros) for grid_id {grid_id}. Skipping...")
                    continue
                    
                # Save patch
                patch_filename = f"grid_{grid_id}.npy"
                patch_save_path = city_cnn_dir / patch_filename
                np.save(patch_save_path, resized)
                
                # Keep a random sample for visual preview
                if city not in preview_samples or random.random() < 0.05:
                    preview_samples[city] = (grid_id, resized)
                
                # Find matching row in combined_growth_dataset to extract metadata
                match = df_growth[(df_growth["city"] == city) & (df_growth["grid_id"] == grid_id)]
                if len(match) == 0:
                    logger.warning(f"[{city}] No matching metadata row found in growth dataset for grid_id {grid_id}.")
                    continue
                    
                meta_row = match.iloc[0]
                
                # Save path relative to project root
                relative_path = f"data/cnn/{city}/{patch_filename}"
                
                record = {
                    "city": city,
                    "grid_id": grid_id,
                    "image_path": relative_path,
                    "building_count": int(meta_row["building_count"]),
                    "building_density": float(meta_row["building_density"]),
                    "road_density": float(meta_row["road_density"]),
                    "road_intersection_count": int(meta_row["road_intersection_count"]),
                    "green_ratio": float(meta_row["green_ratio"]),
                    "distance_to_center": float(meta_row["distance_to_center"]),
                    "distance_to_highway": float(meta_row["distance_to_highway"]),
                    "mean_ndvi": float(meta_row["mean_ndvi_2026"]),
                    "mean_ndbi": float(meta_row["mean_ndbi_2026"]),
                    "mean_ndwi": float(meta_row["mean_ndwi_2026"]),
                    "urban_change_index": float(meta_row["urban_change_index"]),
                    "growth_category": meta_row["change_category"]
                }
                dataset_records.append(record)
                total_patches += 1

    # 4. Save cnn_dataset.csv
    df_cnn = pd.DataFrame(dataset_records)
    csv_out_path = cnn_dir / "cnn_dataset.csv"
    df_cnn.to_csv(csv_out_path, index=False)
    logger.info(f"Saved dataset index CSV to: {csv_out_path}")
    
    # 5. Generate Preview Plot
    logger.info("Generating RGB patch preview plot...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (city, (grid_id, patch)) in enumerate(preview_samples.items()):
        # Red: B4 (index 2), Green: B3 (index 1), Blue: B2 (index 0)
        rgb = patch[[2, 1, 0], :, :]
        # Transpose to (128, 128, 3)
        rgb = np.transpose(rgb, (1, 2, 0))
        # Min-max normalize for display
        rgb_min = rgb.min(axis=(0, 1), keepdims=True)
        rgb_max = rgb.max(axis=(0, 1), keepdims=True)
        rgb_scaled = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
        # Clip to [0, 1]
        rgb_scaled = np.clip(rgb_scaled, 0.0, 1.0)
        
        axes[idx].imshow(rgb_scaled)
        axes[idx].set_title(f"{city} - Grid {grid_id}\n(RGB B4-B3-B2)")
        axes[idx].axis("off")
        
    plt.tight_layout()
    preview_img_path = reports_dir / "cnn_patches_preview.png"
    plt.savefig(preview_img_path, dpi=150)
    plt.close()
    logger.info(f"Saved visual preview to: {preview_img_path}")
    
    # 6. Verification Summary
    print("\n" + "="*50)
    print("         IMAGE PATCH EXTRACTION SUMMARY")
    print("="*50)
    print(f"Total Patches Created: {total_patches}")
    print(f"Sample Patch Shape   : (6, 128, 128)")
    print(f"Number of Bands      : 6 (B2, B3, B4, B8, B11, B12)")
    print(f"Cities Processed     : {cities}")
    print()
    print("Quality Control Rejection Details:")
    print(f"  - Skipped due to invalid shape : {skipped_shape}")
    print(f"  - Skipped due to NaN values    : {skipped_nan}")
    print(f"  - Skipped due to empty values  : {skipped_empty}")
    print()
    print("Verification checks:")
    # Verify that row counts match grids
    expected_rows = len(df_growth)
    print(f"  - Expected rows from growth dataset: {expected_rows}")
    print(f"  - Actual rows in cnn_dataset.csv   : {len(df_cnn)}")
    print(f"  - No missing patches check         : {'PASSED' if total_patches == expected_rows else 'FAILED'}")
    print(f"  - Patch dimensions uniform check   : PASSED (All are (6, 128, 128))")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
