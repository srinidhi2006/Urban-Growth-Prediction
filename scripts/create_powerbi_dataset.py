import os
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# Import config structure
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config

def min_max_scale(series):
    """Normalizes a series using min-max scaling to 0-1 range."""
    s_min = series.min()
    s_max = series.max()
    if s_max - s_min == 0:
        return series * 0.0
    return (series - s_min) / (s_max - s_min)

def main():
    logger.info("Initializing Power BI reporting dataset generation...")
    
    # 1. Setup paths
    features_dir = Config.FEATURES_DIR
    predictions_base_dir = Config.PROJECT_ROOT / "results" / "predictions"
    cache_dir = Config.PROJECT_ROOT / "data" / "cache"
    output_dir = Config.PROJECT_ROOT / "data" / "powerbi"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    combined_features_path = features_dir / "combined_growth_dataset.csv"
    if not combined_features_path.exists():
        logger.error(f"Combined features dataset not found at: {combined_features_path}")
        sys.exit(1)
        
    # Load primary combined growth features
    logger.info(f"Loading primary features dataset: {combined_features_path}")
    df_features = pd.read_csv(combined_features_path)
    df_features["grid_id"] = df_features["grid_id"].astype(np.int64)
    
    cities = ["Bengaluru", "Hyderabad", "Pune"]
    merged_city_dfs = []
    
    for city in cities:
        logger.info(f"Processing city: {city}")
        
        # Load prediction results
        pred_path = predictions_base_dir / city / "prediction_results.csv"
        if not pred_path.exists():
            logger.error(f"Prediction results not found for {city} at: {pred_path}")
            continue
        df_preds = pd.read_csv(pred_path)
        df_preds["grid_id"] = df_preds["grid_id"].astype(np.int64)
        
        # Load locality cache lookup
        cache_path = cache_dir / f"{city.lower()}_localities.csv"
        if not cache_path.exists():
            logger.error(f"Locality cache lookup not found for {city} at: {cache_path}")
            continue
        df_locs = pd.read_csv(cache_path)
        df_locs["grid_id"] = df_locs["grid_id"].astype(np.int64)
        
        # Merge predictions and localities cache
        logger.info(f"Merging predictions and locality cache for {city}")
        df_city_info = pd.merge(df_preds, df_locs, on="grid_id", how="left")
        df_city_info["locality_name"] = df_city_info["locality_name"].fillna("Unknown Locality")
        df_city_info["city"] = city
        
        # Merge with matching primary features subset
        df_features_city = df_features[df_features["city"] == city]
        df_merged_city = pd.merge(df_features_city, df_city_info, on=["city", "grid_id"], how="inner")
        
        logger.info(f"Resolved {len(df_merged_city)} cells for {city}")
        merged_city_dfs.append(df_merged_city)
        
    if not merged_city_dfs:
        logger.error("No city datasets merged. Exiting.")
        sys.exit(1)
        
    df_master = pd.concat(merged_city_dfs, ignore_index=True)
    
    # 2. Derive Business-Friendly Columns
    logger.info("Computing derived business columns...")
    
    # Growth Status based on Growth Score quartiles
    df_master["Growth_Status"] = pd.qcut(
        df_master["growth_score"], 
        q=4, 
        labels=["Low", "Medium", "High", "Very High"], 
        duplicates="drop"
    )
    
    # Environmental Trend based on Delta NDVI
    df_master["Environmental_Trend"] = np.where(
        df_master["delta_ndvi"] > 0.02, "Improving",
        np.where(df_master["delta_ndvi"] < -0.02, "Degrading", "Stable")
    )
    
    # Infrastructure Trend based on Delta NDBI
    df_master["Infrastructure_Trend"] = np.where(
        df_master["delta_ndbi"] > 0.05, "Rapid Expansion",
        np.where((df_master["delta_ndbi"] > 0.01) & (df_master["delta_ndbi"] <= 0.05), "Moderate Expansion", "Stable")
    )
    
    # Priority Level combining Growth Score, Green Ratio, Distance to Highway, Road Density, and Prediction Probability
    norm_growth = min_max_scale(df_master["growth_score"])
    norm_green = 1.0 - min_max_scale(df_master["green_ratio"])  # low green = high priority
    norm_dist_hwy = 1.0 - min_max_scale(df_master["distance_to_highway"]) # close to hwy = high priority
    norm_road_dens = min_max_scale(df_master["road_density"])
    norm_prob = min_max_scale(df_master["prediction_probability"])
    
    priority_score = (
        0.30 * norm_growth +
        0.20 * norm_green +
        0.20 * norm_dist_hwy +
        0.15 * norm_road_dens +
        0.15 * norm_prob
    )
    df_master["Priority_Level"] = pd.qcut(
        priority_score, 
        q=4, 
        labels=["Low", "Moderate", "High", "Critical"], 
        duplicates="drop"
    )
    
    # Development Suitability based on Green Ratio, Road Density, and Distance to Center
    norm_green_suit = min_max_scale(df_master["green_ratio"])
    norm_road_suit = min_max_scale(df_master["road_density"])
    norm_dist_center = 1.0 - min_max_scale(df_master["distance_to_center"]) # closer to center is suitable
    
    suitability_score = (
        0.40 * norm_green_suit +
        0.30 * norm_road_suit +
        0.30 * norm_dist_center
    )
    df_master["Development_Suitability"] = pd.qcut(
        suitability_score, 
        q=4, 
        labels=["Poor", "Average", "Good", "Excellent"], 
        duplicates="drop"
    )
    
    # 3. Data Quality Checks & Column Mapping
    logger.info("Applying data quality rules and formatting column names...")
    
    # Drop duplicates
    initial_rows = len(df_master)
    df_master.drop_duplicates(inplace=True)
    df_master.drop_duplicates(subset=["city", "grid_id"], keep="first", inplace=True)
    post_dedup_rows = len(df_master)
    if initial_rows - post_dedup_rows > 0:
        logger.warning(f"Removed {initial_rows - post_dedup_rows} duplicate rows.")
        
    # Assert non-null coordinate and predictions
    df_master.dropna(subset=["latitude", "longitude", "growth_score", "predicted_growth_category", "prediction_probability"], inplace=True)
    
    # Mapping columns to requested exact layout
    col_mapping = {
        "city": "City",
        "locality_name": "Locality",
        "grid_id": "Grid_ID",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "growth_score": "Growth_Score",
        "predicted_growth_category": "Predicted_Growth_Category",
        "prediction_probability": "Prediction_Probability",
        "building_count": "Building_Count",
        "building_density": "Building_Density",
        "building_area_ratio": "Building_Area_Ratio",
        "road_length": "Road_Length",
        "road_density": "Road_Density",
        "road_intersection_count": "Road_Intersection_Count",
        "intersection_density": "Intersection_Density",
        "green_area": "Green_Area",
        "green_ratio": "Green_Ratio",
        "distance_to_center": "Distance_To_Center",
        "distance_to_highway": "Distance_To_Highway",
        "mean_ndvi_2019": "Mean_NDVI_2019",
        "mean_ndvi_2026": "Mean_NDVI_2026",
        "mean_ndbi_2019": "Mean_NDBI_2019",
        "mean_ndbi_2026": "Mean_NDBI_2026",
        "mean_ndwi_2019": "Mean_NDWI_2019",
        "mean_ndwi_2026": "Mean_NDWI_2026",
        "delta_ndvi": "Delta_NDVI",
        "delta_ndbi": "Delta_NDBI",
        "delta_ndwi": "Delta_NDWI",
        "urban_change_index": "Urban_Change_Index"
    }
    
    df_master.rename(columns=col_mapping, inplace=True)
    
    # 4. Sorting and Rounding
    final_cols = [
        "City", "Locality", "Grid_ID", "Latitude", "Longitude",
        "Growth_Score", "Predicted_Growth_Category", "Prediction_Probability",
        "Building_Count", "Building_Density", "Building_Area_Ratio",
        "Road_Length", "Road_Density", "Road_Intersection_Count", "Intersection_Density",
        "Green_Area", "Green_Ratio", "Distance_To_Center", "Distance_To_Highway",
        "Mean_NDVI_2019", "Mean_NDVI_2026", "Mean_NDBI_2019", "Mean_NDBI_2026",
        "Mean_NDWI_2019", "Mean_NDWI_2026", "Delta_NDVI", "Delta_NDBI", "Delta_NDWI",
        "Urban_Change_Index", "Growth_Status", "Environmental_Trend",
        "Infrastructure_Trend", "Priority_Level", "Development_Suitability"
    ]
    
    df_master = df_master[final_cols]
    
    # Round all numeric fields to 4 decimal places
    numeric_cols = df_master.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        if c != "Grid_ID":
            df_master[c] = df_master[c].round(4)
            
    # Save output
    output_path = output_dir / "powerbi_dataset.csv"
    df_master.to_csv(output_path, index=False)
    logger.success(f"Master Power BI dataset successfully saved to: {output_path}")
    
    # 5. Console Reporting Output
    total_rows = len(df_master)
    rows_by_city = df_master["City"].value_counts().to_dict()
    num_localities = df_master["Locality"].nunique()
    growth_dist = df_master["Predicted_Growth_Category"].value_counts().to_dict()
    avg_growth_score = df_master["Growth_Score"].mean()
    
    # Top metrics listings
    top_10_growth = df_master.sort_values(by="Growth_Score", ascending=False).head(10)[["City", "Locality", "Growth_Score"]].to_dict('records')
    top_10_green = df_master.sort_values(by="Green_Ratio", ascending=False).head(10)[["City", "Locality", "Green_Ratio"]].to_dict('records')
    top_10_priority = df_master[df_master["Priority_Level"] == "Critical"].sort_values(by="Growth_Score", ascending=False).head(10)[["City", "Locality", "Growth_Score"]].to_dict('records')
    if len(top_10_priority) < 10:
        top_10_priority = df_master.sort_values(by="Growth_Score", ascending=False).head(10)[["City", "Locality", "Growth_Score"]].to_dict('records')

    print("\n" + "="*50)
    print("POWER BI CONSOLIDATED DATASET METRIC REPORT")
    print("="*50)
    print(f"Total Rows: {total_rows}")
    print(f"Rows per City: {rows_by_city}")
    print(f"Number of Unique Localities: {num_localities}")
    print(f"Growth Category Distribution: {growth_dist}")
    print(f"Overall Average Growth Score (UCI): {avg_growth_score:.4f}")
    
    print("\nTop 10 Localities by Growth Score:")
    for idx, r in enumerate(top_10_growth):
        print(f"  {idx+1}. {r['Locality']} ({r['City']}) - Score: {r['Growth_Score']:.4f}")
        
    print("\nTop 10 Localities with Highest Green Ratio:")
    for idx, r in enumerate(top_10_green):
        print(f"  {idx+1}. {r['Locality']} ({r['City']}) - Green Ratio: {r['Green_Ratio']:.4f}")
        
    print("\nTop 10 Development Priority Areas:")
    for idx, r in enumerate(top_10_priority):
        print(f"  {idx+1}. {r['Locality']} ({r['City']}) - Score: {r['Growth_Score']:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
