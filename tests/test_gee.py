"""
Unit tests for Earth Engine authentication, boundary loader, and Sentinel-2 pipeline.
"""

from unittest.mock import patch, MagicMock
import pytest
import ee
from config import Config
from gee.utils import get_city_boundary, GAUL_CITY_MAPPING
from gee.sentinel_pipeline import mask_s2_clouds_advanced

# Mock ee.Initialize and ee.Authenticate so GEE won't crash tests if uninitialized
@pytest.fixture(autouse=True)
def mock_ee_init():
    with patch("ee.Initialize"), patch("ee.Authenticate"):
        yield

def test_satellite_config_loading():
    """Verifies Config successfully parses values from config/satellite.yaml."""
    assert Config.SATELLITE_COLLECTION == "COPERNICUS/S2_SR_HARMONIZED"
    assert Config.SATELLITE_CLOUD_PROBABILITY_COLLECTION == "COPERNICUS/S2_CLOUD_PROBABILITY"
    assert Config.SATELLITE_CLOUD_PROBABILITY_THRESHOLD == 60
    assert Config.SATELLITE_CLOUD_PERCENTAGE == 20
    assert Config.SATELLITE_SCALE == 10
    assert Config.SATELLITE_EXPORT_CRS == "EPSG:4326"
    assert "B2" in Config.SATELLITE_BANDS
    assert Config.QC_MIN_CLOUD_FREE_RATIO == 0.3
    assert "B8" in Config.QC_EXPECTED_BANDS

def test_get_city_boundary_local_success():
    """Asserts that loading local GeoJSON geometries for target cities works and yields ee.Geometry."""
    # We patch GEE's Geometry initialization to return a mock or run directly
    with patch("gee.utils.load_geojson_geometry") as mock_load:
        mock_geom = MagicMock(spec=ee.Geometry)
        mock_load.return_value = mock_geom
        
        geom = get_city_boundary("Bengaluru")
        assert geom == mock_geom
        mock_load.assert_called_once()

def test_get_city_boundary_fail_fast():
    """Asserts that searching for an unknown city name raises a ValueError immediately."""
    with pytest.raises(ValueError) as exc_info:
        get_city_boundary("NonExistentCity")
    assert "No valid boundary geometry found for city: NonExistentCity" in str(exc_info.value)

@patch("ee.Image")
@patch("ee.Algorithms.If", create=True)
def test_mask_s2_clouds_bands(mock_if, mock_image_class):
    """Verifies that mask_s2_clouds_advanced calls select, get, and updateMask on target image."""
    mock_image = MagicMock()
    mock_qa = MagicMock()
    mock_prob_img = MagicMock()
    mock_prob = MagicMock()
    mock_final_mask = MagicMock()
    
    # Configure mock chains
    mock_image.select.return_value = mock_qa
    mock_image.propertyNames.return_value.contains.return_value = True
    mock_image.get.return_value = mock_prob_img
    mock_prob_img.select.return_value = mock_prob
    
    mock_if.return_value = mock_final_mask
    mock_image_class.return_value = mock_final_mask
    mock_image.updateMask.return_value = mock_image
    
    masked_img = mask_s2_clouds_advanced(mock_image)
    
    # Assert methods called
    mock_image.select.assert_called_once_with("QA60")
    mock_image.updateMask.assert_called_once_with(mock_final_mask)
    assert masked_img == mock_image

def test_gaul_mapping_names():
    """Asserts GAUL mappings are correctly registered for the target cities."""
    assert GAUL_CITY_MAPPING["Bengaluru"] == "Bangalore Urban"
    assert GAUL_CITY_MAPPING["Hyderabad"] == "Hyderabad"
    assert GAUL_CITY_MAPPING["Pune"] == "Pune"
