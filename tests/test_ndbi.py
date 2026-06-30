"""
Unit tests for the Sentinel-2 NDBI Ingestion pipeline.
Verifies NDBI equations accuracy, NaN division checks, and raster integrity.
"""

import shutil
from pathlib import Path
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from sentinel.ndbi import generate_ndbi
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
    """Helper to generate a small 6-band mosaic TIFF for testing NDBI."""
    # Custom values to verify:
    # Pixel (0,0): NIR=300.0, SWIR=500.0 -> NDBI = 200/800 = 0.25
    # Pixel (0,1): NIR=200.0, SWIR=200.0 -> NDBI = 0/400 = 0.0
    # Pixel (1,0): NIR=0.0, SWIR=0.0 -> NDBI = NaN (no data mask)
    # Pixel (1,1): NIR=500.0, SWIR=300.0 -> NDBI = -200/800 = -0.25
    
    nir_band = np.array([[300.0, 200.0], [0.0, 500.0]], dtype=dtype)
    swir_band = np.array([[500.0, 200.0], [0.0, 300.0]], dtype=dtype)
    dummy_band = np.zeros((2, 2), dtype=dtype)
    
    # Pack into 6 bands (Band 1: B2, Band 2: B3, Band 3: Red(B4), Band 4: NIR(B8), Band 5: B11, Band 6: B12)
    # NIR is Band 4, SWIR is Band 5
    data = np.stack([dummy_band, dummy_band, dummy_band, nir_band, swir_band, dummy_band])
    
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

def test_ndbi_calculation(tmp_path, monkeypatch):
    # Override Config paths to write to temporary folders
    test_proc_dir = tmp_path / "processed"
    
    monkeypatch.setattr(Config, "PROCESSED_DIR", test_proc_dir)
    
    city = "TestCity"
    year = 2026
    
    # Create source synthetic mosaic
    mosaic_path = test_proc_dir / "sentinel" / city / str(year) / f"{city}_{year}_mosaic.tif"
    create_synthetic_mosaic(mosaic_path)
    
    # Run NDBI generation
    success = generate_ndbi(city, year)
    assert success is True
    
    # Verify outputs exist
    ndbi_path = test_proc_dir / "sentinel" / city / str(year) / f"{city}_{year}_NDBI.tif"
    preview_path = test_proc_dir / "sentinel" / city / str(year) / f"{city}_{year}_NDBI_preview.png"
    
    assert ndbi_path.exists()
    assert preview_path.exists()
    
    # Open and verify the NDBI TIFF math
    with rasterio.open(ndbi_path) as src:
        assert src.count == 1
        assert src.height == 2
        assert src.width == 2
        assert src.crs.to_epsg() == 3857
        assert src.dtypes[0] == "float32"
        
        ndbi_data = src.read(1)
        # Expected outputs:
        # (0,0): 0.25
        # (0,1): 0.0
        # (1,0): NaN (due to 0 values)
        # (1,1): -0.25
        assert np.isclose(ndbi_data[0, 0], 0.25)
        assert np.isclose(ndbi_data[0, 1], 0.0)
        assert np.isnan(ndbi_data[1, 0])
        assert np.isclose(ndbi_data[1, 1], -0.25)
