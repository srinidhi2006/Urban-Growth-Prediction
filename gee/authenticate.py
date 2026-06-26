"""
Google Earth Engine Authentication and Initialization Module.
Provides secure authentication setup supporting service account credentials or local user tokens.
"""

import ee
from loguru import logger
from config import Config

def authenticate_gee() -> bool:
    """Authenticates and initializes the Google Earth Engine Python API.
    
    Checks environment variables and service account key files in Config before falling back
    to system-level OAuth credentials.
    
    Returns:
        bool: True if authentication and initialization succeeded, False otherwise.
    """
    logger.info("Initializing Google Earth Engine authentication...")
    
    # 1. Attempt Service Account Authentication
    if Config.GEE_SERVICE_ACCOUNT_EMAIL and Config.GEE_SERVICE_ACCOUNT_KEY_PATH:
        key_path = Config.PROJECT_ROOT / Config.GEE_SERVICE_ACCOUNT_KEY_PATH
        if key_path.exists():
            try:
                logger.info(f"Authenticating via Service Account: {Config.GEE_SERVICE_ACCOUNT_EMAIL}")
                credentials = ee.ServiceAccountCredentials(
                    Config.GEE_SERVICE_ACCOUNT_EMAIL,
                    str(key_path)
                )
                ee.Initialize(
                    credentials=credentials,
                    project=Config.GEE_PROJECT_ID if Config.GEE_PROJECT_ID else None
                )
                logger.success("Earth Engine initialized successfully using Service Account.")
                return True
            except Exception as e:
                logger.error(f"Service Account authentication failed: {e}")
        else:
            logger.warning(f"Service account key file not found at: {key_path}. Attempting local defaults...")
            
    # 2. Fallback to Local User Credentials
    try:
        logger.info("Initializing Earth Engine using local system/OAuth default credentials...")
        if Config.GEE_PROJECT_ID:
            ee.Initialize(project=Config.GEE_PROJECT_ID)
        else:
            ee.Initialize()
        logger.success("Earth Engine initialized successfully using default credentials.")
        return True
    except Exception as e:
        logger.error(
            f"Local system credential initialization failed: {e}\n"
            "Please run 'earthengine authenticate' on the command line or configure a Service Account."
        )
        return False

if __name__ == "__main__":
    # Test script utility
    success = authenticate_gee()
    if success:
        print("GEE initialized successfully.")
    else:
        print("GEE initialization failed.")
