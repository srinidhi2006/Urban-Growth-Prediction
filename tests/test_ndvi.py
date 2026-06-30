"""
Unit tests for the Sentinel-2 NDVI Ingestion pipeline.
Verifies NDVI equations accuracy, NaN division checks, metadata stats, and validation check reports.
"""

import json
import shutil
from pathlib import Path
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from sentinel.ndvi import generate_ndvi
from sentinel.validator import verify_ndvi
from config import Config

# Mock GEE initialization during tests
@pytest.fixture(autouse=True)
def mock_ee_init(monkeypatch):
    monkeypatch.setattr("ee.Initialize", lambda: None)
    monkeypatch.setattr("ee.Authenticate", lambda: None)

def create_synthetic_mosaic(
    path: Path,
    crs: str = "EPSG:3857",
    res: float = 10.0,
    dtype: str = "float32"
):
    """Helper to generate a small 6-band mosaic TIFF for testing NDVI."""
    # Custom values to verify:
    # Pixel (0,0): Red=100.0, NIR=300.0 -> NDVI = 200/400 = 0.5
    # Pixel (0,1): Red=200.0, NIR=200.0 -> NDVI = 0/400 = 0.0
    # Pixel (1,0): Red=0.0, NIR=0.0 -> NDVI = NaN (no data mask)
    # Pixel (1,1): Red=500.0, NIR=500.0 -> NDVI = 0/1000 = 0.0
    
    red_band = np.array([[100.0, 200.0], [0.0, 500.0]], dtype=dtype)
    nir_band = np.array([[300.0, 200.0], [0.0, 500.0]], dtype=dtype)
    dummy_band = np.zeros((2, 2), dtype=dtype)
    
    # Pack into 6 bands (Band 1: B2, Band 2: B3, Band 3: Red(B4), Band 4: NIR(B8), Band 5: B11, Band 6: B12)
    # Band 3 is index 2, Band 4 is index 3
    data = np.stack([dummy_band, dummy_band, red_band, nir_band, dummy_band, dummy_band])
    
    transform = from_origin(0.0, 100.0, res, res)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=2,
        width=2,
        count=6,
        dtype=dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data)

def test_ndvi_calculation_and_validation(tmp_path, monkeypatch):
    # Override Config paths to write to temporary folders
    test_proc_dir = tmp_path / "processed"
    test_feat_dir = tmp_path / "features"
    
    monkeypatch.setattr(Config, "PROCESSED_DIR", test_proc_dir)
    monkeypatch.setattr(Config, "FEATURES_DIR", test_feat_dir)
    
    city = "TestCity"
    year = 2026
    
    # Create source synthetic mosaic
    mosaic_path = test_proc_dir / "sentinel" / city / str(year) / f"{city}_{year}_mosaic.tif"
    create_synthetic_mosaic(mosaic_path)
    
    # Run NDVI generation
    success = generate_ndvi(city, year)
    assert success is True
    
    # Verify outputs exist
    ndvi_path = test_proc_dir / "sentinel" / city / str(year) / f"{city}_{year}_NDVI.tif"
    preview_path = test_proc_dir / "sentinel" / city / str(year) / f"{city}_{year}_NDVI_preview.png"
    metadata_path = test_proc_dir / "sentinel" / city / str(year) / "ndvi_metadata.json"
    
    assert ndvi_path.exists()
    assert preview_path.exists()
    assert metadata_path.exists()
    
    # Open and verify the NDVI TIFF math
    with rasterio.open(ndvi_path) as src:
        assert src.count == 1
        assert src.height == 2
        assert src.width == 2
        assert src.crs.to_epsg() == 3857
        assert src.dtypes[0] == "float32"
        
        ndvi_data = src.read(1)
        # Expected outputs:
        # (0,0): 0.5
        # (0,1): 0.0
        # (1,0): NaN (due to 0 values)
        # (1,1): 0.0
        assert np.isclose(ndvi_data[0, 0], 0.5)
        assert np.isclose(ndvi_data[0, 1], 0.0)
        assert np.isnan(ndvi_data[1, 0])
        assert np.isclose(ndvi_data[1, 1], 0.0)
        
    # Verify metadata contents
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["city"] == city
    assert meta["year"] == year
    assert meta["formula"] == "(B8-B4)/(B8+B4)"
    assert meta["band_red"] == "B4"
    assert meta["band_nir"] == "B8"
    assert meta["minimum_ndvi"] == 0.0
    assert meta["maximum_ndvi"] == 0.5
    assert meta["mean_ndvi"] == pytest.approx(0.1667, abs=1e-4) # (0.5 + 0.0 + 0.0) / 3 = 0.16667
    assert meta["valid_pixels"] == 3
    assert meta["nodata_pixels"] == 1
    
    # Verify validation report
    valid = verify_ndvi(city, year, ndvi_path, mosaic_path)
    assert valid is True
    
    report_path = test_feat_dir / f"ndvi_validation_{city}_{year}.json"
    assert report_path.exists()
    
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["passed"] is True
    assert report["value_range_check"] is True
    assert report["datatype_check"] is True
    assert report["crs_check"] is True
    assert report["transform_check"] is True
    assert report["nodata_check"] is True
