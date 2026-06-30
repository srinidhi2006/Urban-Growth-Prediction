"""
Unit tests for the Sentinel Tiling module.
"""

import shutil
import json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
from sentinel.tiling import generate_sentinel_tiles
from sentinel.validator import validate_tiles
from config import Config

def test_generate_sentinel_tiles():
    # Setup test city name and synthetic geometries
    city = "TestCity"
    
    # Clean up any leftover folders before test
    interim_dir = Config.INTERIM_DIR / "sentinel" / city
    feat_dir = Config.FEATURES_DIR
    if interim_dir.exists():
        shutil.rmtree(interim_dir)
        
    # A 12,000m x 12,000m square starting at (0, 0) in EPSG:3857 coordinates
    poly = box(0, 0, 12000, 12000)
    gdf_3857 = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:3857")
    
    # Project to WGS84 (EPSG:4326) as input boundary
    gdf_wgs84 = gdf_3857.to_crs("EPSG:4326")
    
    # Execute tile generation with tile_size = 5000m
    # 5000m size: np.arange(0, 12000, 5000) -> [0, 5000, 10000] (3 divisions in X and Y -> 9 tiles)
    success = generate_sentinel_tiles(city, gdf_wgs84, tile_size_meters=5000)
    
    assert success is True
    
    # Verify outputs
    tiles_geojson_path = interim_dir / "tiles.geojson"
    summary_path = interim_dir / "tile_summary.json"
    boundaries_dir = interim_dir / "tile_boundaries"
    
    assert tiles_geojson_path.exists()
    assert summary_path.exists()
    assert boundaries_dir.exists()
    
    # Load tiles layer
    tiles_gdf = gpd.read_file(tiles_geojson_path)
    assert len(tiles_gdf) == 9
    assert str(tiles_gdf.crs).upper() == "EPSG:3857"
    
    # Assert columns structure and values
    required_cols = ["tile_id", "city", "tile_name", "status", "geometry"]
    for col in required_cols:
        assert col in tiles_gdf.columns
        
    assert (tiles_gdf["city"] == city).all()
    assert (tiles_gdf["status"] == "pending").all()
    assert list(tiles_gdf["tile_id"]) == list(range(9))
    assert tiles_gdf["tile_name"].iloc[0] == "TestCity_tile_000"
    
    # Verify summary JSON
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["city"] == city
    assert summary["tile_size_m"] == 5000
    assert summary["tiles_generated"] == 9
    assert "crs" in summary
    assert "generated_at" in summary
    assert "boundary_area_sqkm" in summary
    
    # Verify individual boundary files exist
    for idx in range(9):
        tile_file = boundaries_dir / f"tile_{idx:03d}.geojson"
        assert tile_file.exists()
        
        # Load and check it contains its row
        single_tile = gpd.read_file(tile_file)
        assert len(single_tile) == 1
        assert int(single_tile["tile_id"].iloc[0]) == idx
        
    # Execute validation
    valid = validate_tiles(city, tiles_gdf, gdf_wgs84)
    assert valid is True
    
    report_file = feat_dir / f"tile_validation_report_{city}.json"
    assert report_file.exists()
    
    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["passed"] is True
    assert report["checks"]["total_tiles_validated"] == 9
    assert report["checks"]["duplicate_tile_ids_found"] == 0
    assert report["checks"]["empty_geometries_found"] == 0
    
    # Clean up test output folders
    if interim_dir.exists():
        shutil.rmtree(interim_dir)
    if report_file.exists():
        report_file.unlink()
