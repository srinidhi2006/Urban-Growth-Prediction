import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from backend.city_analysis import analyze_city, download_boundary

@pytest.fixture
def mock_xgboost_model():
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 2])
    model.predict_proba.return_value = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.05, 0.9]
    ])
    return model

@pytest.fixture
def mock_scaler():
    scaler = MagicMock()
    scaler.transform.side_effect = lambda x: x
    return scaler

@pytest.fixture
def mock_shap_values():
    # Shape should match (samples=3, features=24, classes=3)
    return np.random.uniform(-0.5, 0.5, (3, 24, 3))

def test_download_boundary_mock():
    with patch("osmnx.geocode_to_gdf") as mock_ox:
        mock_gdf = MagicMock()
        mock_gdf.empty = False
        mock_ox.return_value = mock_gdf
        
        output_path = Path("temp_test_boundary.geojson")
        success = download_boundary("TestCity", output_path)
        
        assert success is True
        mock_ox.assert_called_once()
        mock_gdf.to_file.assert_called_once_with(output_path, driver="GeoJSON")

@patch("pickle.load")
@patch("builtins.open")
@patch("pandas.read_csv")
@patch("pathlib.Path.exists")
@patch("shap.TreeExplainer")
def test_analyze_city_cached(mock_explainer, mock_exists, mock_read_csv, mock_open, mock_pickle_load, mock_xgboost_model, mock_scaler, mock_shap_values):
    # Setup mocks
    # We mock path check for model/scaler files and cached features CSV
    mock_exists.side_effect = lambda: True
    
    # 24 expected features
    mock_features = pd.DataFrame({
        "grid_id": [0, 1, 2],
        "building_count": [10, 20, 30],
        "building_area_ratio": [0.1, 0.2, 0.3],
        "road_length": [1000, 2000, 3000],
        "distance_to_highway": [500, 1000, 1500],
        "green_area": [200, 400, 600],
        "distance_to_center": [5000, 4000, 3000],
        "mean_ndvi_2019": [0.2, 0.3, 0.4],
        "mean_ndbi_2019": [-0.1, 0.0, 0.1],
        "mean_ndwi_2019": [-0.5, -0.4, -0.3],
        "mean_ndvi_2026": [0.15, 0.25, 0.35],
        "mean_ndbi_2026": [0.0, 0.1, 0.2],
        "mean_ndwi_2026": [-0.55, -0.45, -0.35],
        "delta_ndvi": [-0.05, -0.05, -0.05],
        "delta_ndbi": [0.1, 0.1, 0.1],
        "delta_ndwi": [-0.05, -0.05, -0.05],
        "abs_delta_ndvi": [0.05, 0.05, 0.05],
        "abs_delta_ndbi": [0.1, 0.1, 0.1],
        "abs_delta_ndwi": [0.05, 0.05, 0.05],
        "norm_delta_ndvi": [0.5, 0.5, 0.5],
        "norm_delta_ndbi": [0.5, 0.5, 0.5],
        "norm_delta_ndwi": [0.5, 0.5, 0.5],
        "urban_change_index": [0.25, 0.5, 0.75],
        "change_category": ["Low", "Medium", "High"]
    })
    mock_read_csv.return_value = mock_features
    
    # Pickle side effects to return scaler then xgboost model
    mock_pickle_load.side_effect = [mock_xgboost_model, mock_scaler]
    
    # Mock SHAP Explainer
    mock_tree = MagicMock()
    mock_tree.shap_values.return_value = mock_shap_values
    mock_explainer.return_value = mock_tree
    
    # Prevent saving files physically by mocking pandas to_csv and json dump
    with patch("pandas.DataFrame.to_csv") as mock_to_csv, patch("json.dump") as mock_json_dump:
        results = analyze_city("TestCity")
        
        # Verify structure of returned dictionary
        assert results["city"] == "TestCity"
        assert results["number_of_grids"] == 3
        assert results["high_growth_count"] == 1
        assert results["medium_growth_count"] == 1
        assert results["low_growth_count"] == 1
        assert results["average_growth_score"] == 0.5
        
        # Verify output files save triggers
        assert mock_to_csv.call_count == 2 # 1 for predictions, 1 for shap values
        assert mock_json_dump.call_count == 1
