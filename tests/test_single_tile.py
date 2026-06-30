"""
Unit tests for Sentinel Single-Tile download, verification, and manifest updating logic.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import geopandas as gpd
from shapely.geometry import box
import ee
from scripts.download_sentinel import update_manifest, write_tile_metadata
from sentinel.validator import verify_downloaded_tile
from config import Config

# Mock GEE initialization during tests
@pytest.fixture(autouse=True)
def mock_ee_init():
    with patch("ee.Initialize"), patch("ee.Authenticate"):
        yield

def test_tile_boundary_loading_and_gee_geometry(tmp_path):
    """Test that a local tile boundary GeoJSON file loads correctly and converts to ee.Geometry."""
    # Create a synthetic tile boundary GeoJSON
    tile_geom = box(0, 0, 5000, 5000)
    tile_gdf = gpd.GeoDataFrame(
        [{"tile_id": 0, "city": "Bengaluru", "tile_name": "Bengaluru_tile_000", "status": "pending"}],
        geometry=[tile_geom],
        crs="EPSG:3857"
    )
    
    geojson_path = tmp_path / "tile_000.geojson"
    tile_gdf.to_file(geojson_path, driver="GeoJSON")
    
    # Verify we can load it and project it
    loaded_gdf = gpd.read_file(geojson_path)
    assert len(loaded_gdf) == 1
    assert loaded_gdf["tile_name"].iloc[0] == "Bengaluru_tile_000"
    
    # Project to WGS84 for GEE conversion
    loaded_wgs84 = loaded_gdf.to_crs("EPSG:4326")
    from shapely.geometry import mapping
    geom_dict = mapping(loaded_wgs84.geometry.iloc[0])
    
    with patch("ee.Geometry") as mock_geom_class:
        mock_ee_geom = MagicMock(spec=ee.Geometry)
        mock_geom_class.return_value = mock_ee_geom
        
        ee_geom = ee.Geometry(geom_dict)
        assert ee_geom == mock_ee_geom
        mock_geom_class.assert_called_once_with(geom_dict)

def test_write_tile_metadata(tmp_path):
    """Tests metadata logging writes all required fields and export stats."""
    meta_path = tmp_path / "tile_000_metadata.json"
    
    qc_metrics = {"valid_pixel_ratio": 0.98}
    verify_results = {"crs": "EPSG:3857", "is_geographic": False, "resolution": 10.0, "width": 500, "height": 500, "bands_count": 6}
    
    # Run metadata writer
    write_tile_metadata(
        meta_path=meta_path,
        city="Bengaluru",
        year=2019,
        tile_id=0,
        tile_name="Bengaluru_tile_000",
        qc_metrics=qc_metrics,
        image_count=12,
        verify_results=verify_results,
        processing_duration=15.4,
        file_size_mb=4.8
    )
    
    assert meta_path.exists()
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    assert meta["city"] == "Bengaluru"
    assert meta["year"] == 2019
    assert meta["tile_id"] == 0
    assert meta["tile_name"] == "Bengaluru_tile_000"
    assert meta["CRS"] == "EPSG:3857"
    assert meta["pixel_resolution_val"] == 10.0
    assert meta["pixel_resolution_unit"] == "meters"
    assert meta["image_dimensions"]["width"] == 500
    assert meta["image_dimensions"]["height"] == 500
    assert meta["gee_collection_size"] == 12
    assert meta["valid_pixel_ratio"] == 0.98
    assert meta["processing_time_sec"] == 15.4
    assert meta["download_size_mb"] == 4.8

def test_update_manifest(tmp_path):
    """Tests that download manifest tracks completed, failed, and pending tiles correctly."""
    manifest_path = tmp_path / "download_manifest.json"
    summary_path = tmp_path / "tile_summary.json"
    
    # Save a mock tile summary with 3 tiles
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump({"tiles_generated": 3}, sf)
        
    # Mark tile 0 as success
    update_manifest(manifest_path, "Bengaluru", 2019, tile_id=0, success=True)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["completed_tiles"] == [0]
    assert manifest["failed_tiles"] == []
    assert manifest["pending_tiles"] == [1, 2]
    
    # Mark tile 1 as failed
    update_manifest(manifest_path, "Bengaluru", 2019, tile_id=1, success=False)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["completed_tiles"] == [0]
    assert manifest["failed_tiles"] == [1]
    assert manifest["pending_tiles"] == [2]

@patch("rasterio.open")
def test_verify_downloaded_tile(mock_rasterio_open, tmp_path):
    """Tests that verification function parses metadata, verifies bands count, and writes validation JSON."""
    mock_src = MagicMock()
    mock_src.width = 500
    mock_src.height = 500
    mock_src.count = 6
    mock_src.res = (10.0, -10.0)
    mock_crs = MagicMock()
    mock_crs.is_geographic = False
    mock_crs.__str__.return_value = "EPSG:3857"
    mock_src.crs = mock_crs
    mock_src.bounds = MagicMock(left=100.0, bottom=200.0, right=300.0, top=400.0)
    
    # Set up context manager mock
    mock_rasterio_open.return_value.__enter__.return_value = mock_src
    
    # Temporary mock tif path
    tif_path = tmp_path / "tile_000.tif"
    tif_path.write_bytes(b"") # Write mock empty file
    
    # Override features folder in Config to write validation report in tmp_path
    features_report = Config.FEATURES_DIR / "tile_download_validation_TestCity_tile000.json"
    if features_report.exists():
        features_report.unlink()
        
    results = verify_downloaded_tile(
        city="TestCity",
        year=2019,
        tile_id=0,
        tile_name="TestCity_tile_000",
        tif_path=tif_path
    )
    
    assert results["width"] == 500
    assert results["height"] == 500
    assert results["bands_count"] == 6
    assert results["resolution"] == 10.0
    assert results["passed"] is True
    
    # Check that report JSON exists
    assert features_report.exists()
    with open(features_report, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    assert report["city"] == "TestCity"
    assert report["tile_name"] == "TestCity_tile_000"
    assert report["download_success"] is True
    assert report["raster_verification"] is True
    assert report["band_count"] == 6
    assert report["crs_check"] is True
    assert report["pixel_size_check"] is True
    
    # Cleanup
    if features_report.exists():
        features_report.unlink()
