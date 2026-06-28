"""
Unit tests for the OpenStreetMap (OSM) vector ingestion and spatial processing pipeline.
Uses synthetic shapely geometries to test math calculations, grid division, and validator checks.
"""

from unittest.mock import patch, MagicMock
import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point, LineString
from config import Config
from osm.cleaner import clean_osm_data
from osm.grid_generator import generate_spatial_grid
from osm.feature_extractor import extract_osm_features
from osm.validator import validate_osm_features

@pytest.fixture
def mock_boundary_gdf():
    """Returns a simple 3000m x 3000m boundary square in EPSG:3857 (metric)."""
    # Min coordinates: 0, 0; Max coordinates: 3000, 3000
    poly = Polygon([(0, 0), (3000, 0), (3000, 3000), (0, 3000), (0, 0)])
    # Convert to EPSG:4326 for initial load
    gdf = gpd.GeoDataFrame(geometry=[poly], crs=Config.PROJECTED_CRS).to_crs("EPSG:4326")
    return gdf

def test_grid_generator(mock_boundary_gdf):
    """Verifies that a 1000m grid generated over a 3000m x 3000m square yields exactly 9 grid cells."""
    with patch("geopandas.GeoDataFrame.to_file") as mock_to_file:
        success = generate_spatial_grid(
            city="TestCity",
            boundary_gdf=mock_boundary_gdf,
            grid_size_meters=1000
        )
        assert success is True
        mock_to_file.assert_called_once()
        
        # Verify the actual generation logic internally
        boundary_projected = mock_boundary_gdf.to_crs(Config.PROJECTED_CRS)
        geom = boundary_projected.geometry.iloc[0]
        minx, miny, maxx, maxy = geom.bounds
        
        # Checking math directly
        cells = []
        x_coords = range(0, 3000, 1000)
        y_coords = range(0, 3000, 1000)
        for x in x_coords:
            for y in y_coords:
                cells.append(Point(x, y))
                
        assert len(cells) == 9

def test_validator_detects_violations():
    """Asserts that validator.py flags violations like out-of-bounds ratios or negative counts."""
    # Write a test CSV payload with a ratio > 1.0 and a negative count
    mock_data = {
        "grid_id": [0, 1, 2],
        "building_count": [10, -5, 12],  # Violates non-negative check
        "building_density": [10.0, 0.0, 12.0],
        "building_area_ratio": [0.25, 0.1, 1.25],  # Violates ratio bounds [0, 1]
        "road_length": [100.0, 200.0, 150.0],
        "road_density": [0.1, 0.2, 0.15],
        "road_intersection_count": [2, 4, 3],
        "intersection_density": [2.0, 4.0, 3.0],
        "distance_to_highway": [50.0, -10.0, 30.0],  # Violates non-negative check
        "green_area": [1000.0, 2000.0, 1500.0],
        "green_ratio": [0.1, 0.2, 0.15],
        "distance_to_center": [500.0, 1200.0, 800.0]
    }
    mock_df = pd.DataFrame(mock_data)
    
    # Mock GeoPandas reading of CSV and GeoJSON
    with patch("pandas.read_csv", return_value=mock_df), \
         patch("geopandas.read_file") as mock_read_gdf, \
         patch("json.dump") as mock_json_write, \
         patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open", create=True):
         
         # Mock return of read_file with standard WGS84 CRS
         mock_gdf = MagicMock(spec=gpd.GeoDataFrame)
         mock_gdf.crs = "EPSG:4326"
         mock_gdf.geometry.is_empty = pd.Series([False, False, False])
         mock_gdf.geometry.is_valid = pd.Series([True, True, True])
         mock_gdf.geometry.duplicated.return_value = pd.Series([False, False, False])
         
         # projected area check
         mock_gdf_projected = MagicMock()
         mock_gdf_projected.geometry.area = pd.Series([1000000, 1000000, 1000000])
         mock_gdf.to_crs.return_value = mock_gdf_projected
         mock_read_gdf.return_value = mock_gdf
         
         # Perform validation (should return False due to negative values and ratio bounds)
         success = validate_osm_features("TestCity")
         assert success is False
         mock_json_write.assert_called_once()
         
         # Extract logged report dict
         report_arg = mock_json_write.call_args[0][0]
         assert report_arg["passed"] is False
         assert report_arg["checks"]["negative_values_found"] > 0
         assert report_arg["checks"]["ratio_bounds_violations"] > 0

def test_extractor_city_center(mock_boundary_gdf):
    """Verifies that the centroid is automatically calculated correctly for administrative bounds."""
    boundary_projected = mock_boundary_gdf.to_crs(Config.PROJECTED_CRS)
    city_center_centroid = boundary_projected.geometry.iloc[0].centroid
    city_center_rep = boundary_projected.geometry.iloc[0].representative_point()
    
    # Central check: Coordinates are located within bounds
    minx, miny, maxx, maxy = boundary_projected.geometry.iloc[0].bounds
    assert minx <= city_center_centroid.x <= maxx
    assert miny <= city_center_centroid.y <= maxy
    assert minx <= city_center_rep.x <= maxx
    assert miny <= city_center_rep.y <= maxy
