"""
CLI Entry Point for the AI-Powered Urban Growth Prediction Platform.
Orchestrates data collection, feature engineering, modeling, explainability, and service deployment.
"""

import argparse
import sys
from loguru import logger
from config import Config

def verify_setup() -> bool:
    """Verifies that the project structure, configuration, and dependencies are correctly set up.
    
    Returns:
        bool: True if verification passes, False otherwise.
    """
    logger.info("Starting project environment and setup verification...")
    success = True
    
    # 1. Verify environment loading
    logger.info(f"Active Environment: {Config.ENV}")
    logger.info(f"Debug Mode: {Config.DEBUG}")
    
    # 2. Check Directory Structure
    required_paths = {
        "Sentinel-2 Raw Data": Config.RAW_SENTINEL_DIR,
        "Dynamic World Raw Data": Config.RAW_DYNAMIC_WORLD_DIR,
        "OSM Raw Data": Config.RAW_OSM_DIR,
        "Interim Data": Config.INTERIM_DIR,
        "Processed Data": Config.PROCESSED_DIR,
        "Spatial Tiles": Config.TILES_DIR,
        "Feature Tables": Config.FEATURES_DIR,
        "Model Artifacts": Config.MODELS_DIR,
        "Scalers": Config.SCALERS_DIR,
        "SHAP Explanations": Config.SHAP_DIR,
        "Metrics Dumps": Config.METRICS_DIR,
        "Logs": Config.LOGS_DIR,
    }
    
    logger.info("--- Directory Verification ---")
    for name, path in required_paths.items():
        if path.exists():
            logger.info(f" [PASS] {name} folder exists at: {path}")
        else:
            logger.error(f" [FAIL] {name} folder is missing: {path}")
            success = False

    # 3. Check Core Dependency Imports (Dynamic Checks)
    logger.info("--- Core Dependency Verification ---")
    dependencies = [
        ("ee", "Google Earth Engine (earthengine-api)"),
        ("geopandas", "GeoPandas"),
        ("rasterio", "Rasterio"),
        ("shapely", "Shapely"),
        ("osmnx", "OSMnx"),
        ("torch", "PyTorch"),
        ("torchvision", "Torchvision"),
        ("xgboost", "XGBoost"),
        ("shap", "SHAP Explainability"),
        ("fastapi", "FastAPI"),
    ]
    
    for module_name, desc in dependencies:
        try:
            __import__(module_name)
            logger.info(f" [PASS] {desc} is available and importable.")
        except ImportError:
            logger.warning(f" [WARN] {desc} ({module_name}) is not installed in the active environment.")
            # We treat dependency warnings as non-fatal for setup structure checks,
            # but log warning for the user's local venv guidance.

    # 4. Check Config Parameters
    logger.info("--- Configuration Checks ---")
    if not Config.GEE_PROJECT_ID:
        logger.warning(" [WARN] GEE_PROJECT_ID environment variable is empty. GEE modules will require this.")
    else:
        logger.info(f" [PASS] GEE Cloud Project configured: '{Config.GEE_PROJECT_ID}'")
        
    if success:
        logger.success("Project structure verification completed successfully.")
    else:
        logger.error("Project structure verification encountered errors. Please check logs.")
        
    return success

def main():
    """Main CLI entry point for command parsing."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Urban Growth Prediction Platform CLI Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core actions
    parser.add_argument(
        "--verify-setup",
        action="store_true",
        help="Run environment diagnostics, directory existence validation, and import test validation"
    )
    
    parser.add_argument(
        "--city",
        type=str,
        choices=Config.CITIES,
        help="Target city for actions (e.g. Bengaluru, Hyderabad, Pune)"
    )
    
    # Sub-stage execution targets (defined but not implemented in Phase 0)
    parser.add_argument(
        "--ingest-data",
        action="store_true",
        help="[Future] Trigger Google Earth Engine and OpenStreetMap download pipelines"
    )
    
    parser.add_argument(
        "--process-features",
        action="store_true",
        help="[Future] Execute spatial feature engineering, calculation of indices and EfficientNet embeddings"
    )
    
    parser.add_argument(
        "--train-models",
        action="store_true",
        help="[Future] Run ML training (Random Forest/XGBoost) and Deep Learning routines"
    )
    
    parser.add_argument(
        "--explain",
        action="store_true",
        help="[Future] Generate SHAP value plots and feature importance reports"
    )

    args = parser.parse_args()

    # CLI Action Routing
    if args.verify_setup:
        setup_ok = verify_setup()
        sys.exit(0 if setup_ok else 1)
        
    elif args.ingest_data or args.process_features or args.train_models or args.explain:
        logger.warning(
            "This action is reserved for subsequent phases. "
            "Please run 'python main.py --verify-setup' to test Phase 0 environment setup."
        )
        sys.exit(0)
        
    else:
        parser.print_help()
        sys.exit(0)

if __name__ == "__main__":
    main()
