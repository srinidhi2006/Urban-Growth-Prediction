import os
import pandas as pd
from pathlib import Path
from loguru import logger
from config import Config

def min_max_scale(series):
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return series * 0.0
    return (series - s_min) / (s_max - s_min)


def compute_growth_score() -> bool:
    try:
        cities = ["Bengaluru", "Hyderabad", "Pune"]
        combined_dfs = []
        
        osm_cols = [
            "building_count", "building_density", "building_area_ratio",
            "road_length", "road_density", "road_intersection_count",
            "intersection_density", "distance_to_highway", "green_area",
            "green_ratio", "distance_to_center"
        ]
        
        for city in cities:
            logger.info(f"Processing growth score for {city}...")
            
            # File paths
            f_2019_path = Config.FEATURES_DIR / f"ml_features_{city}_2019.csv"
            f_2026_path = Config.FEATURES_DIR / f"ml_features_{city}_2026.csv"
            
            if not f_2019_path.exists():
                logger.error(f"2019 features file not found for {city}: {f_2019_path}")
                return False
            if not f_2026_path.exists():
                logger.error(f"2026 features file not found for {city}: {f_2026_path}")
                return False
                
            df_2019 = pd.read_csv(f_2019_path)
            df_2026 = pd.read_csv(f_2026_path)
            
            # Rename spectral columns
            df_2019_renamed = df_2019[["grid_id", "mean_ndvi", "mean_ndbi", "mean_ndwi"]].rename(
                columns={
                    "mean_ndvi": "mean_ndvi_2019",
                    "mean_ndbi": "mean_ndbi_2019",
                    "mean_ndwi": "mean_ndwi_2019"
                }
            )
            
            df_2026_renamed = df_2026[["grid_id", "mean_ndvi", "mean_ndbi", "mean_ndwi"]].rename(
                columns={
                    "mean_ndvi": "mean_ndvi_2026",
                    "mean_ndbi": "mean_ndbi_2026",
                    "mean_ndwi": "mean_ndwi_2026"
                }
            )
            
            # Extract OSM features (take from 2019 dataset)
            df_osm = df_2019[["grid_id"] + osm_cols]
            
            # Join datasets
            df_city = df_osm.merge(df_2019_renamed, on="grid_id").merge(df_2026_renamed, on="grid_id")
            
            # Compute Deltas
            df_city["delta_ndvi"] = df_city["mean_ndvi_2026"] - df_city["mean_ndvi_2019"]
            df_city["delta_ndbi"] = df_city["mean_ndbi_2026"] - df_city["mean_ndbi_2019"]
            df_city["delta_ndwi"] = df_city["mean_ndwi_2026"] - df_city["mean_ndwi_2019"]
            
            # Compute Absolute Deltas
            df_city["abs_delta_ndvi"] = df_city["delta_ndvi"].abs()
            df_city["abs_delta_ndbi"] = df_city["delta_ndbi"].abs()
            df_city["abs_delta_ndwi"] = df_city["delta_ndwi"].abs()
            
            # Compute Normalized Deltas (within city)
            df_city["norm_delta_ndvi"] = min_max_scale(-df_city["delta_ndvi"])
            df_city["norm_delta_ndbi"] = min_max_scale(df_city["delta_ndbi"])
            df_city["norm_delta_ndwi"] = min_max_scale(-df_city["delta_ndwi"])
            
            # Compute Urban Change Index
            raw_uci = (
                0.5 * df_city["norm_delta_ndbi"] +
                0.3 * df_city["norm_delta_ndvi"] +
                0.2 * df_city["norm_delta_ndwi"]
            )
            df_city["urban_change_index"] = min_max_scale(raw_uci)
            
            # Compute Change Category using quantile-based classification within each city separately
            q33 = df_city["urban_change_index"].quantile(0.33)
            q66 = df_city["urban_change_index"].quantile(0.66)
            
            def get_quantile_category(val):
                if val <= q33:
                    return "Low"
                elif val <= q66:
                    return "Medium"
                else:
                    return "High"
            
            df_city["change_category"] = df_city["urban_change_index"].apply(get_quantile_category)
            
            # Ensure correct column ordering
            output_cols = (
                ["grid_id"] + osm_cols +
                ["mean_ndvi_2019", "mean_ndbi_2019", "mean_ndwi_2019"] +
                ["mean_ndvi_2026", "mean_ndbi_2026", "mean_ndwi_2026"] +
                ["delta_ndvi", "delta_ndbi", "delta_ndwi"] +
                ["abs_delta_ndvi", "abs_delta_ndbi", "abs_delta_ndwi"] +
                ["norm_delta_ndvi", "norm_delta_ndbi", "norm_delta_ndwi"] +
                ["urban_change_index", "change_category"]
            )
            df_city = df_city[output_cols]
            
            # Save city-specific growth dataset
            output_city_path = Config.FEATURES_DIR / f"{city.lower()}_growth_dataset.csv"
            df_city.to_csv(output_city_path, index=False)
            logger.info(f"Saved {city} growth dataset to {output_city_path}")
            
            # Prepare for combined dataframe
            df_city_combined = df_city.copy()
            df_city_combined.insert(0, "city", city)
            combined_dfs.append(df_city_combined)
            
        # Concatenate and save combined dataset
        df_combined = pd.concat(combined_dfs, ignore_index=True)
        output_combined_path = Config.FEATURES_DIR / "combined_growth_dataset.csv"
        df_combined.to_csv(output_combined_path, index=False)
        logger.info(f"Saved combined growth dataset to {output_combined_path}")
        
        return True
    except Exception as e:
        logger.exception(f"Error computing growth score: {e}")
        return False
