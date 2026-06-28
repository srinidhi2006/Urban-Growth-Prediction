"""
Configuration Manager for the AI-Powered Urban Growth Prediction Platform.
Handles loading environment variables, defining project paths, configuring logging,
and setting placeholder configurations for future modules.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env if it exists
load_dotenv()

class Config:
    """Project configuration loader and directory manager."""
    
    # --- Project Root & Environment ---
    PROJECT_ROOT: Path = Path(__file__).resolve().parent
    ENV: str = os.getenv("ENV", "development").lower()
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
    
    # --- Data Path Specifications ---
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    RAW_SENTINEL_DIR: Path = RAW_DIR / "sentinel"
    RAW_DYNAMIC_WORLD_DIR: Path = RAW_DIR / "dynamic_world"
    RAW_OSM_DIR: Path = RAW_DIR / "osm"
    INTERIM_DIR: Path = DATA_DIR / "interim"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    TILES_DIR: Path = DATA_DIR / "tiles"
    FEATURES_DIR: Path = DATA_DIR / "features"
    BOUNDARIES_DIR: Path = PROJECT_ROOT / "assets" / "boundaries"
    
    # --- Artifact Path Specifications ---
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    MODELS_DIR: Path = ARTIFACTS_DIR / "models"
    SCALERS_DIR: Path = ARTIFACTS_DIR / "scalers"
    SHAP_DIR: Path = ARTIFACTS_DIR / "shap"
    METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"
    
    # --- Logs ---
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    LOG_FILE_PATH: Path = LOGS_DIR / "platform.log"
    
    # --- GEE Settings ---
    GEE_PROJECT_ID: str = os.getenv("GEE_PROJECT_ID", "")
    GEE_SERVICE_ACCOUNT_EMAIL: str = os.getenv("GEE_SERVICE_ACCOUNT_EMAIL", "")
    GEE_SERVICE_ACCOUNT_KEY_PATH: str = os.getenv("GEE_SERVICE_ACCOUNT_KEY_PATH", "")
    
    # --- Model Hyperparameters ---
    DEFAULT_TRAIN_TEST_SPLIT: float = float(os.getenv("DEFAULT_TRAIN_TEST_SPLIT", "0.8"))
    EFFICIENTNET_BATCH_SIZE: int = int(os.getenv("EFFICIENTNET_BATCH_SIZE", "32"))
    NUM_WORKERS: int = int(os.getenv("NUM_WORKERS", "4"))
    
    # --- FastAPI Configs ---
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_DEBUG: bool = os.getenv("API_DEBUG", "True").lower() == "true"
    
    # --- OCI Configuration Placeholders ---
    OCI_CONFIG_PROFILE: str = os.getenv("OCI_CONFIG_PROFILE", "DEFAULT")
    OCI_COMPARTMENT_ID: str = os.getenv("OCI_COMPARTMENT_ID", "")
    OCI_DATA_SCIENCE_PROJECT_ID: str = os.getenv("OCI_DATA_SCIENCE_PROJECT_ID", "")
    OCI_BUCKET_NAME: str = os.getenv("OCI_BUCKET_NAME", "")
    OCI_MODEL_DEPLOYMENT_OCID: str = os.getenv("OCI_MODEL_DEPLOYMENT_OCID", "")
    
    # --- Geospatial Parameters (Historical Baseline) ---
    TIMELINE_YEARS: list = [2019, 2026]
    CITIES: list = ["Bengaluru", "Hyderabad", "Pune"]
    
    # Coordinate Reference Systems (CRS) standard defaults
    GLOBAL_CRS: str = "EPSG:4326"       # WGS84 for storage/API coordinates
    PROJECTED_CRS: str = "EPSG:3857"    # Web Mercator for meter-based distance density metrics

    # --- Load satellite.yaml settings ---
    _satellite_config_path = PROJECT_ROOT / "config" / "satellite.yaml"
    _sat_config = {}
    if _satellite_config_path.exists():
        try:
            with open(_satellite_config_path, "r") as f:
                _sat_config = yaml.safe_load(f).get("sentinel", {})
        except Exception as e:
            print(f"Warning: Failed to load satellite.yaml config: {e}")

    SATELLITE_COLLECTION: str = _sat_config.get("collection", "COPERNICUS/S2_SR_HARMONIZED")
    SATELLITE_CLOUD_PROBABILITY_COLLECTION: str = _sat_config.get("cloud_probability_collection", "COPERNICUS/S2_CLOUD_PROBABILITY")
    SATELLITE_CLOUD_PROBABILITY_THRESHOLD: int = int(_sat_config.get("cloud_probability_threshold", 60))
    SATELLITE_CLOUD_PERCENTAGE: int = int(_sat_config.get("cloud_percentage", 20))
    SATELLITE_SCALE: int = int(_sat_config.get("scale", 10))
    SATELLITE_EXPORT_CRS: str = _sat_config.get("export_crs", "EPSG:4326")
    SATELLITE_BANDS: list = _sat_config.get("bands", ["B2", "B3", "B4", "B8", "B11", "B12"])
    
    QC_MIN_CLOUD_FREE_RATIO: float = float(_sat_config.get("quality_check", {}).get("min_cloud_free_ratio", 0.3))
    QC_EXPECTED_RESOLUTION: int = int(_sat_config.get("quality_check", {}).get("expected_resolution", 10))
    QC_EXPECTED_BANDS: list = _sat_config.get("quality_check", {}).get("expected_bands", ["B2", "B3", "B4", "B8", "B11", "B12"])

    # --- Load osm.yaml settings ---
    _osm_config_path = PROJECT_ROOT / "config" / "osm.yaml"
    _osm_config = {}
    if _osm_config_path.exists():
        try:
            with open(_osm_config_path, "r") as f:
                _osm_config = yaml.safe_load(f).get("osm", {})
        except Exception as e:
            print(f"Warning: Failed to load osm.yaml config: {e}")

    OSM_TIMEOUT: int = int(_osm_config.get("timeout", 180))
    OSM_MAX_RETRIES: int = int(_osm_config.get("max_retries", 5))
    OSM_GRID_SIZE_METERS: int = int(_osm_config.get("grid_size_meters", 1000))
    OSM_TAGS_BUILDINGS: dict = _osm_config.get("tags", {}).get("buildings", {"building": True})
    OSM_TAGS_GREEN_AREAS: dict = _osm_config.get("tags", {}).get("green_areas", {})
    OSM_TAGS_ROADS: dict = _osm_config.get("tags", {}).get("roads", {"highway": True})
    OSM_MAJOR_HIGHWAYS: list = _osm_config.get("major_highways", ["motorway", "trunk", "primary", "secondary"])

    @classmethod
    def initialize_directories(cls):
        """Creates any missing local directory structures required by the project."""
        paths = [
            cls.RAW_SENTINEL_DIR,
            cls.RAW_DYNAMIC_WORLD_DIR,
            cls.RAW_OSM_DIR,
            cls.INTERIM_DIR,
            cls.PROCESSED_DIR,
            cls.TILES_DIR,
            cls.FEATURES_DIR,
            cls.BOUNDARIES_DIR,
            cls.MODELS_DIR,
            cls.SCALERS_DIR,
            cls.SHAP_DIR,
            cls.METRICS_DIR,
            cls.LOGS_DIR
        ]
        
        for path in paths:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Initialized project folder path: {path}")

    @classmethod
    def configure_logging(cls):
        """Sets up loguru logger rotation, formats, and sinks."""
        # Ensure log folder exists
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Configure Loguru standard output
        logger.remove()  # Remove default handler
        
        # Stream logger
        logger.add(
            sink=lambda msg: print(msg, end=""),
            level="DEBUG" if cls.DEBUG else "INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )
        
        # File logger
        logger.add(
            sink=cls.LOG_FILE_PATH,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
        logger.info("Logging infrastructure configured successfully.")

# Auto-initialize paths and log config when module is loaded
Config.configure_logging()
Config.initialize_directories()
