"""
OSM Feature Extractor Stage.
Aggregates clipped geometries to grid cells, computes automatically detected city center,
and engineers 11 final infrastructure features for machine learning.
"""

import time
import pandas as pd
import geopandas as gpd
import osmnx as ox
from loguru import logger
from config import Config

def extract_osm_features(city: str, boundary_gdf: gpd.GeoDataFrame) -> bool:
    """Aggregates spatial datasets, calculates metrics, and creates the ML-ready dataset.
    
    Args:
        city (str): Target city name.
        boundary_gdf (gpd.GeoDataFrame): Administrative boundary (EPSG:4326).
        
    Returns:
        bool: True if feature extraction completed successfully, False otherwise.
    """
    logger.info(f"Extracting 11 ML-ready infrastructure features for {city}...")
    start_time = time.time()
    
    interim_city_dir = Config.INTERIM_DIR / "osm" / city
    raw_city_dir = Config.RAW_OSM_DIR / city
    
    # Define file paths
    grid_path = interim_city_dir / "grid.geojson"
    buildings_path = interim_city_dir / "buildings_joined.geojson"
    green_path = interim_city_dir / "green_areas_joined.geojson"
    roads_path = interim_city_dir / "roads_joined.geojson"
    roads_cleaned_path = interim_city_dir / "roads_cleaned.geojson"
    graphml_path = raw_city_dir / "roads.graphml"
    
    # Output paths
    features_dir = Config.FEATURES_DIR
    features_dir.mkdir(parents=True, exist_ok=True)
    
    csv_out_path = features_dir / f"osm_features_{city}.csv"
    geojson_out_path = features_dir / f"osm_features_{city}.geojson"
    
    try:
        # Load grid
        grid = gpd.read_file(grid_path)
        grid_len = len(grid)
        
        # Load joined layers
        buildings_joined = gpd.read_file(buildings_path)
        green_joined = gpd.read_file(green_path)
        roads_joined = gpd.read_file(roads_path)
        roads_cleaned = gpd.read_file(roads_cleaned_path)
        
        # Calculate cell area
        grid["cell_area_m2"] = grid.geometry.area
        grid["cell_area_km2"] = grid["cell_area_m2"] / 1_000_000.0
        
        # 1. Building Count, Density, and Area Ratio
        logger.info(f"[{city}] Extracting building features...")
        b_counts = buildings_joined.groupby("grid_id").size().reindex(grid["grid_id"], fill_value=0)
        
        # Compute building area inside each cell (Sum areas of pre-clipped buildings)
        if not buildings_joined.empty:
            b_areas = buildings_joined.groupby("grid_id").apply(
                lambda x: x.geometry.area.sum(),
                include_groups=False
            )
            b_areas = b_areas.reindex(grid["grid_id"], fill_value=0.0)
        else:
            b_areas = pd.Series(0.0, index=grid["grid_id"])
            
        grid["building_count"] = b_counts.values
        grid["building_density"] = grid["building_count"] / grid["cell_area_km2"]
        grid["building_area_ratio"] = b_areas.values / grid["cell_area_m2"]
        
        # 2. Road Length and Road Density
        logger.info(f"[{city}] Extracting road features...")
        if not roads_joined.empty:
            r_lengths = roads_joined.groupby("grid_id").apply(lambda x: x.geometry.length.sum(), include_groups=False)
            r_lengths = r_lengths.reindex(grid["grid_id"], fill_value=0.0)
        else:
            r_lengths = pd.Series(0.0, index=grid["grid_id"])
            
        grid["road_length"] = r_lengths.values
        grid["road_density"] = (grid["road_length"] / 1000.0) / grid["cell_area_km2"]
        
        # 3. Intersections and Intersection Density (From raw GraphML)
        logger.info(f"[{city}] Extracting intersection junctions...")
        intersections_gdf = gpd.GeoDataFrame(geometry=[], crs=Config.PROJECTED_CRS)
        
        if graphml_path.exists():
            try:
                graph = ox.load_graphml(str(graphml_path))
                nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False)
                
                # Intersections are where street_count > 2
                if "street_count" in nodes.columns:
                    intersections_gdf = nodes[nodes["street_count"] > 2].to_crs(Config.PROJECTED_CRS)
                else:
                    # Fallback
                    intersections_gdf = nodes.to_crs(Config.PROJECTED_CRS)
            except Exception as e:
                logger.warning(f"[{city}] Failed to parse intersection nodes from GraphML: {e}")
                
        if not intersections_gdf.empty:
            # Join intersections to grid
            joined_intersections = gpd.sjoin(intersections_gdf, grid, how="inner", predicate="intersects")
            intersection_counts = joined_intersections.groupby("grid_id").size().reindex(grid["grid_id"], fill_value=0)
        else:
            intersection_counts = pd.Series(0, index=grid["grid_id"])
            
        grid["road_intersection_count"] = intersection_counts.values
        grid["intersection_density"] = grid["road_intersection_count"] / grid["cell_area_km2"]
        
        # 4. Distance to Nearest Major Highway
        logger.info(f"[{city}] Extracting distance to nearest major highway...")
        
        def is_major_highway(val):
            if not val:
                return False
            if isinstance(val, list):
                return any(item in Config.OSM_MAJOR_HIGHWAYS for item in val)
            return val in Config.OSM_MAJOR_HIGHWAYS
            
        if not roads_cleaned.empty and "highway" in roads_cleaned.columns:
            major_highways = roads_cleaned[roads_cleaned["highway"].apply(is_major_highway)]
        else:
            major_highways = gpd.GeoDataFrame(geometry=[], crs=Config.PROJECTED_CRS)
            
        # Compute grid cell centroids
        centroids = gpd.GeoDataFrame(
            grid[["grid_id"]], 
            geometry=grid.geometry.centroid, 
            crs=Config.PROJECTED_CRS
        )
        
        if not major_highways.empty:
            # Spatial join to the nearest major highway
            joined_dist = gpd.sjoin_nearest(centroids, major_highways, distance_col="dist_to_hwy", how="left")
            dist_map = joined_dist.groupby("grid_id")["dist_to_hwy"].min().reindex(grid["grid_id"], fill_value=999999.0)
        else:
            dist_map = pd.Series(999999.0, index=grid["grid_id"])
            
        grid["distance_to_highway"] = dist_map.values
        
        # 5. Green Area and Green Ratio (Sum areas of green polygons)
        logger.info(f"[{city}] Extracting green space metrics...")
        if not green_joined.empty:
            g_areas = green_joined.groupby("grid_id").apply(
                lambda x: x.geometry.area.sum(),
                include_groups=False
            )
            g_areas = g_areas.reindex(grid["grid_id"], fill_value=0.0)
        else:
            g_areas = pd.Series(0.0, index=grid["grid_id"])
            
        grid["green_area"] = g_areas.values
        grid["green_ratio"] = grid["green_area"] / grid["cell_area_m2"]
        
        # 6. Distance to Centroid City Center (Automatic detection via representative_point)
        logger.info(f"[{city}] Extracting distance to automatic city center...")
        boundary_projected = boundary_gdf.to_crs(Config.PROJECTED_CRS)
        city_center = boundary_projected.geometry.iloc[0].representative_point()
        
        grid["distance_to_center"] = grid.geometry.centroid.distance(city_center)
        
        # Cleanup temporary metrics columns before saving
        final_grid = grid.drop(columns=["cell_area_m2", "cell_area_km2"])
        
        # Save to CSV (convert geometries to WGS84 coordinates for future integration)
        logger.info(f"[{city}] Saving ML-ready tabular features CSV...")
        final_grid_wgs84 = final_grid.to_crs("EPSG:4326")
        
        # Exclude geometry column for standard machine-learning-ready CSV format
        # but keep grid_id as key index.
        csv_df = pd.DataFrame(final_grid_wgs84.drop(columns=["geometry"]))
        csv_df.to_csv(csv_out_path, index=False)
        
        # Save to GeoJSON (with EPSG:4326 coordinates for visualization engines)
        logger.info(f"[{city}] Saving visual features GeoJSON...")
        final_grid_wgs84.to_file(geojson_out_path, driver="GeoJSON")
        
        duration = time.time() - start_time
        logger.success(f"[{city}] Extraction complete in {duration:.2f}s. Saved to: {csv_out_path.name}")
        return True
        
    except Exception as e:
        logger.exception(f"[{city}] Extraction stage crashed: {e}")
        return False
