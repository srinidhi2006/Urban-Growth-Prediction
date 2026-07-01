import os
import json
import pandas as pd
import geopandas as gpd
import osmnx as ox
from pathlib import Path
from loguru import logger

# Add project root to path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config

def main():
    logger.info("Initializing locality caches generation...")
    cache_dir = Config.PROJECT_ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cities = ["Bengaluru", "Hyderabad", "Pune"]
    
    for city in cities:
        logger.info(f"Ingesting localities for: {city}...")
        
        # Load grid GeoJSON to get centroids and IDs
        geojson_path = Config.FEATURES_DIR / f"osm_features_{city}.geojson"
        if not geojson_path.exists():
            logger.error(f"Grid GeoJSON not found for {city} at: {geojson_path}")
            continue
            
        grid_gdf = gpd.read_file(geojson_path)
        
        # Compute centroids in EPSG:4326
        centroids = grid_gdf.geometry.centroid
        latitudes = centroids.y
        longitudes = centroids.x
        
        # Geocode place names from OSMnx
        boundary_path = Config.BOUNDARIES_DIR / f"{city}.geojson"
        boundary_gdf = gpd.read_file(boundary_path)
        geom = boundary_gdf.geometry.iloc[0]
        
        try:
            places = ox.features_from_polygon(geom, {'place': ['suburb', 'neighbourhood', 'locality', 'village', 'town', 'quarter']})
            if not places.empty and "name" in places.columns:
                places_clean = places[["name", "geometry"]].dropna().to_crs(Config.PROJECTED_CRS)
                grid_projected = grid_gdf.to_crs(Config.PROJECTED_CRS)
                centroids_proj = grid_projected.geometry.centroid
                
                locality_names = []
                for cent in centroids_proj:
                    distances = places_clean.geometry.distance(cent)
                    nearest_idx = distances.idxmin()
                    locality_names.append(places_clean.loc[nearest_idx, "name"])
            else:
                locality_names = [f"Sector {i+1}" for i in range(len(grid_gdf))]
        except Exception as e:
            logger.warning(f"OSMnx query failed for {city}: {e}. Generating default names.")
            locality_names = [f"Sector {i+1}" for i in range(len(grid_gdf))]
            
        # Build DataFrame
        df_cache = pd.DataFrame({
            "grid_id": grid_gdf["grid_id"].astype(str),
            "latitude": latitudes,
            "longitude": longitudes,
            "locality_name": locality_names
        })
        
        output_path = cache_dir / f"{city.lower()}_localities.csv"
        df_cache.to_csv(output_path, index=False)
        logger.success(f"Saved cached localities for {city} to: {output_path}")

if __name__ == "__main__":
    main()
