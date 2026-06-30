"""
Unit tests for the Sentinel-2 Mosaic Ingestion pipeline.
Verifies raster merging, pre-merge consistency verification, validation logic, and metadata formatting.
"""

import json
import shutil
from pathlib import Path
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from sentinel.mosaic import create_mosaic
from sentinel.validator import verify_mosaic
from config import Config

# Mock GEE initialization during tests
@pytest.fixture(autouse=True)
def mock_ee_init(monkeypatch):
    monkeypatch.setattr("ee.Initialize", lambda: None)
    monkeypatch.setattr("ee.Authenticate", lambda: None)

def create_synthetic_tile(
    path: Path,
    crs: str = "EPSG:3857",
    res: float = 10.0,
    count: int = 6,
    dtype: str = "float32",
    origin_x: float = 0.0,
    origin_y: float = 100.0
):
    """Helper to generate a small 2x2 synthetic TIFF raster for testing."""
    transform = from_origin(origin_x, origin_y, res, res)
    data = np.ones((count, 2, 2), dtype=dtype) * 1000.0 # standard reflection value
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=2,
        width=2,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data)

def test_mosaic_creation_and_validation(tmp_path, monkeypatch):
    # Override Config paths to write to temporary folders
    test_raw_dir = tmp_path / "raw"
    test_proc_dir = tmp_path / "processed"
    test_feat_dir = tmp_path / "features"
    
    monkeypatch.setattr(Config, "RAW_SENTINEL_DIR", test_raw_dir)
    monkeypatch.setattr(Config, "PROCESSED_DIR", test_proc_dir)
    monkeypatch.setattr(Config, "FEATURES_DIR", test_feat_dir)
    
    city = "TestCity"
    year = 2026
    
    # 1. Create two adjacent synthetic tiles (Tile 0 and Tile 1)
    tile0_path = test_raw_dir / city / str(year) / "tiles" / "tile_000.tif"
    tile1_path = test_raw_dir / city / str(year) / "tiles" / "tile_001.tif"
    
    # Tile 0 is from X=0 to 20, Y=80 to 100 (res=10m)
    create_synthetic_tile(tile0_path, origin_x=0.0, origin_y=100.0)
    # Tile 1 is from X=20 to 40, Y=80 to 100 (res=10m, touching right border of Tile 0)
    create_synthetic_tile(tile1_path, origin_x=20.0, origin_y=100.0)
    
    # Run mosaic creation
    success = create_mosaic(city, year)
    assert success is True
    
    # Verify outputs exist
    mosaic_path = test_proc_dir / "sentinel" / city / str(year) / f"{city}_{year}_mosaic.tif"
    preview_path = test_proc_dir / "sentinel" / city / str(year) / f"{city}_{year}_preview.png"
    metadata_path = test_proc_dir / "sentinel" / city / str(year) / "mosaic_metadata.json"
    
    assert mosaic_path.exists()
    assert preview_path.exists()
    assert metadata_path.exists()
    
    # Open and verify the merged TIFF
    with rasterio.open(mosaic_path) as src:
        assert src.count == 6
        assert src.height == 2
        assert src.width == 4  # merged horizontally (2px + 2px = 4px width)
        assert src.crs.to_epsg() == 3857
        assert src.res == (10.0, 10.0)
        
    # Verify metadata contents
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["city"] == city
    assert meta["year"] == year
    assert meta["tiles_merged"] == 2
    assert meta["bands"] == 6
    assert meta["width"] == 4
    assert meta["height"] == 2
    assert meta["crs"] == "EPSG:3857"
    assert meta["pixel_size"] == 10.0
    assert meta["pixel_size_unit"] == "meters"
    
    # Verify mosaic validator report
    valid = verify_mosaic(city, year, mosaic_path, tiles_count=2)
    assert valid is True
    
    report_path = test_feat_dir / f"mosaic_validation_{city}_{year}.json"
    assert report_path.exists()
    
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["passed"] is True
    assert report["tiles_merged"] == 2
    assert report["band_count"] == 6
    assert report["crs_check"] is True
    assert report["resolution_check"] is True
    assert report["raster_integrity"] is True

def test_mosaic_consistency_mismatch_checks(tmp_path, monkeypatch):
    test_raw_dir = tmp_path / "raw"
    test_proc_dir = tmp_path / "processed"
    
    monkeypatch.setattr(Config, "RAW_SENTINEL_DIR", test_raw_dir)
    monkeypatch.setattr(Config, "PROCESSED_DIR", test_proc_dir)
    
    city = "TestCity"
    year = 2026
    
    tiles_dir = test_raw_dir / city / str(year) / "tiles"
    
    # Mismatch Case A: CRS Mismatch
    shutil.rmtree(test_raw_dir, ignore_errors=True)
    create_synthetic_tile(tiles_dir / "tile_000.tif", crs="EPSG:3857")
    create_synthetic_tile(tiles_dir / "tile_001.tif", crs="EPSG:4326")
    success = create_mosaic(city, year)
    assert success is False
    
    # Mismatch Case B: Band Count Mismatch
    shutil.rmtree(test_raw_dir, ignore_errors=True)
    create_synthetic_tile(tiles_dir / "tile_000.tif", count=6)
    create_synthetic_tile(tiles_dir / "tile_001.tif", count=3)
    success = create_mosaic(city, year)
    assert success is False
    
    # Mismatch Case C: Resolution Mismatch
    shutil.rmtree(test_raw_dir, ignore_errors=True)
    create_synthetic_tile(tiles_dir / "tile_000.tif", res=10.0)
    create_synthetic_tile(tiles_dir / "tile_001.tif", res=30.0)
    success = create_mosaic(city, year)
    assert success is False
    
    # Mismatch Case D: Data Type Mismatch
    shutil.rmtree(test_raw_dir, ignore_errors=True)
    create_synthetic_tile(tiles_dir / "tile_000.tif", dtype="float32")
    create_synthetic_tile(tiles_dir / "tile_001.tif", dtype="uint16")
    success = create_mosaic(city, year)
    assert success is False
