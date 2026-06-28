"""
OSM Downloader Stage.
Queries OpenStreetMap via OSMnx for buildings, road network, and green spaces,
saving raw geospatial layers and download stats.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
import geopandas as gpd
import osmnx as ox
from loguru import logger
from config import Config

# Configure OSMnx settings from global configuration
ox.settings.timeout = Config.OSM_TIMEOUT
ox.settings.max_retries = Config.OSM_MAX_RETRIES
ox.settings.use_cache = True
ox.settings.log_console = True

def _execute_with_endpoint_rotation(query_func, *args, **kwargs):
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://z.overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter"
    ]
    
    last_err = None
    for endpoint in endpoints:
        ox.settings.overpass_endpoint = endpoint
        logger.debug(f"Overpass endpoint set to: {endpoint}")
        
        for attempt in range(1, Config.OSM_MAX_RETRIES + 1):
            try:
                return query_func(*args, **kwargs)
            except Exception as e:
                last_err = e
                err_msg = str(e).lower()
                
                # If the error is indicating that the query returned no elements,
                # we immediately stop retrying and raise the exception, as this is a
                # logical outcome (no features exist) rather than a server failure.
                if "empty" in err_msg or "no elements" in err_msg:
                    raise e
                    
                wait_time = attempt * 5
                logger.warning(
                    f"OSM query failed on attempt {attempt}/{Config.OSM_MAX_RETRIES} "
                    f"using {endpoint}: {e}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                
    raise last_err

def download_osm_data(city: str, boundary_gdf: gpd.GeoDataFrame) -> bool:
    """Downloads buildings, green spaces, and road networks from OSM for a given city boundary.
    
    Args:
        city (str): Name of the target city.
        boundary_gdf (gpd.GeoDataFrame): GeoDataFrame containing the city boundary (EPSG:4326).
        
    Returns:
        bool: True if downloading and stats writing completed successfully, False otherwise.
    """
    logger.info(f"Starting raw OSM data download for {city}...")
    start_time = time.time()
    
    # 1. Verify directory exists
    raw_city_dir = Config.RAW_OSM_DIR / city
    raw_city_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract boundary polygon (ensure WGS84 EPSG:4326)
    if boundary_gdf.crs != "EPSG:4326":
        logger.info(f"Re-projecting boundary to EPSG:4326 for OSM querying...")
        boundary_gdf = boundary_gdf.to_crs("EPSG:4326")
    
    boundary_geom = boundary_gdf.geometry.iloc[0]
    
    try:
        # 2. Download building footprints
        logger.info(f"[{city}] Querying building features...")
        try:
            # Try features_from_polygon (OSMnx >= 1.9.0)
            if hasattr(ox, "features_from_polygon"):
                buildings = _execute_with_endpoint_rotation(ox.features_from_polygon, boundary_geom, tags=Config.OSM_TAGS_BUILDINGS)
            else:
                buildings = _execute_with_endpoint_rotation(ox.geometries_from_polygon, boundary_geom, tags=Config.OSM_TAGS_BUILDINGS)
        except Exception as e:
            logger.warning(f"[{city}] Buildings fetch returned empty or failed: {e}. Creating empty layer.")
            buildings = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            
        # 3. Download green spaces
        logger.info(f"[{city}] Querying green area features...")
        try:
            if hasattr(ox, "features_from_polygon"):
                green_areas = _execute_with_endpoint_rotation(ox.features_from_polygon, boundary_geom, tags=Config.OSM_TAGS_GREEN_AREAS)
            else:
                green_areas = _execute_with_endpoint_rotation(ox.geometries_from_polygon, boundary_geom, tags=Config.OSM_TAGS_GREEN_AREAS)
        except Exception as e:
            logger.warning(f"[{city}] Green spaces fetch returned empty or failed: {e}. Creating empty layer.")
            green_areas = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            
        # 4. Download road network graph
        logger.info(f"[{city}] Querying road network graph...")
        try:
            graph = _execute_with_endpoint_rotation(ox.graph_from_polygon, boundary_geom, network_type="all")
            # Convert graph edges to a GeoDataFrame
            _, roads = ox.graph_to_gdfs(graph, nodes=True, edges=True)
        except Exception as e:
            logger.warning(f"[{city}] Roads graph fetch returned empty or failed: {e}. Creating empty layers.")
            graph = None
            roads = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

            
        # 5. Save Raw Geometries
        logger.info(f"[{city}] Saving raw geospatial datasets...")
        
        buildings_path = raw_city_dir / "buildings.geojson"
        buildings.to_file(buildings_path, driver="GeoJSON")
        
        green_path = raw_city_dir / "green_areas.geojson"
        green_areas.to_file(green_path, driver="GeoJSON")
        
        roads_path = raw_city_dir / "roads.geojson"
        roads.to_file(roads_path, driver="GeoJSON")
        
        if graph is not None:
            graph_path = raw_city_dir / "roads.graphml"
            ox.save_graphml(graph, filepath=str(graph_path))
            logger.info(f"[{city}] Saved roads graphml network representation.")
            
        # 6. Generate and save download_statistics.json
        building_count = len(buildings)
        green_polygons = len(green_areas)
        road_segments = len(roads)
        
        stats = {
            "city": city,
            "building_count": building_count,
            "road_segments": road_segments,
            "green_polygons": green_polygons,
            "download_timestamp": datetime.utcnow().isoformat() + "Z",
            "pipeline_version": "1.2.0",
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }
        
        stats_path = raw_city_dir / "download_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
            
        duration = time.time() - start_time
        logger.success(
            f"[{city}] OSM Download complete in {duration:.2f}s. "
            f"Buildings: {building_count}, Roads: {road_segments}, Green Spaces: {green_polygons}"
        )
        return True
        
    except Exception as e:
        logger.exception(f"[{city}] Downloader crashed: {e}")
        return False
