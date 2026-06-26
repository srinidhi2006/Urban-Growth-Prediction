"""
Unit tests for the Config class and directory auto-initialization properties.
"""

from pathlib import Path
from config import Config

def test_config_paths():
    """Asserts that all defined path properties are Path objects and resolve correctly."""
    assert isinstance(Config.PROJECT_ROOT, Path)
    assert isinstance(Config.DATA_DIR, Path)
    assert isinstance(Config.RAW_SENTINEL_DIR, Path)
    assert isinstance(Config.INTERIM_DIR, Path)
    assert isinstance(Config.PROCESSED_DIR, Path)
    assert isinstance(Config.MODELS_DIR, Path)
    assert isinstance(Config.LOGS_DIR, Path)

def test_directory_existence():
    """Asserts that the initialization routine successfully created the directories."""
    Config.initialize_directories()
    
    assert Config.RAW_SENTINEL_DIR.exists()
    assert Config.RAW_DYNAMIC_WORLD_DIR.exists()
    assert Config.RAW_OSM_DIR.exists()
    assert Config.INTERIM_DIR.exists()
    assert Config.PROCESSED_DIR.exists()
    assert Config.TILES_DIR.exists()
    assert Config.FEATURES_DIR.exists()
    assert Config.MODELS_DIR.exists()
    assert Config.SCALERS_DIR.exists()
    assert Config.SHAP_DIR.exists()
    assert Config.METRICS_DIR.exists()
    assert Config.LOGS_DIR.exists()

def test_default_constants():
    """Asserts that configuration parameters match baseline requirements."""
    assert "Bengaluru" in Config.CITIES
    assert "Hyderabad" in Config.CITIES
    assert "Pune" in Config.CITIES
    assert 2019 in Config.TIMELINE_YEARS
    assert 2026 in Config.TIMELINE_YEARS
    assert Config.SATELLITE_SCALE == 10
    assert Config.GLOBAL_CRS == "EPSG:4326"
    assert Config.PROJECTED_CRS == "EPSG:3857"
