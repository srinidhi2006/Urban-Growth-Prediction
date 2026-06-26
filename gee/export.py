"""
Google Earth Engine Export and Local Downloader Module.
Coordinates direct raster downloads, GEE tasks creation, preview generation, and metadata logging.
"""

import io
import json
import time
import zipfile
import subprocess
from pathlib import Path
import ee
import requests
from loguru import logger
from config import Config

PIPELINE_VERSION = "1.1.0"

def get_git_commit() -> str:
    """Retrieves the short Git commit hash of the current HEAD.
    
    Returns:
        str: Git commit hash, or descriptive string if git is unavailable.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return commit
    except FileNotFoundError:
        return "git_not_installed"
    except subprocess.CalledProcessError:
        return "not_a_git_repo"
    except Exception:
        return "unknown"

def requests_get_with_retry(
    url: str, 
    stream: bool = False, 
    timeout: int = 120, 
    max_retries: int = 3, 
    initial_delay: int = 2
) -> requests.Response:
    """Sends an HTTP GET request with exponential backoff retry logic.
    
    Args:
        url (str): Target HTTP URL.
        stream (bool): Stream response parameter.
        timeout (int): Maximum connection timeout in seconds.
        max_retries (int): Number of retries before failing.
        initial_delay (int): Delay in seconds before the first retry.
        
    Returns:
        requests.Response: Response object.
        
    Raises:
        requests.RequestException: If the final retry attempt fails.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"HTTP GET Attempt {attempt}/{max_retries}...")
            response = requests.get(url, stream=stream, timeout=timeout)
            if response.status_code >= 500:
                response.raise_for_status()
            return response
        except requests.RequestException as e:
            if attempt == max_retries:
                logger.error(f"HTTP GET failed after {max_retries} attempts: {e}")
                raise e
            logger.warning(f"Request failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2


def generate_rgb_preview(image: ee.Image, region: ee.Geometry, output_path: Path) -> bool:
    """Generates and saves a stretched 3-channel RGB visual PNG preview from GEE.
    
    Uses B4 (Red), B3 (Green), and B2 (Blue) bands and stretches values (0-3000)
    for optimal visual display.
    
    Args:
        image (ee.Image): GEE Image composite.
        region (ee.Geometry): Clip geometry.
        output_path (Path): Filepath to write the preview (.png).
        
    Returns:
        bool: True if success, False otherwise.
    """
    logger.info(f"Generating RGB PNG preview image for: {output_path.name}")
    try:
        # Scale for preview can be slightly coarser (e.g. 30m or 50m) to save time and bandwidth
        preview_scale = max(30, int(Config.SATELLITE_SCALE * 2))
        
        # Configure visualization parameters (reflection values normally 0-10000, stretched to 0-3000)
        vis_params = {
            "region": region,
            "scale": preview_scale,
            "crs": Config.SATELLITE_EXPORT_CRS,
            "format": "png",
            "min": 0,
            "max": 3000,
            "bands": ["B4", "B3", "B2"]
        }
        
        url = image.getThumbURL(vis_params)
        
        response = requests_get_with_retry(url, timeout=60)
        if response.status_code == 200:
            output_path.write_bytes(response.content)
            logger.success(f"RGB preview saved successfully at: {output_path}")
            return True
        else:
            logger.error(f"Failed to fetch thumb preview URL. GEE returned code {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error generating PNG preview: {e}")
        return False

def write_metadata(
    output_path: Path,
    city: str,
    year: int,
    image_count: int,
    qc_metrics: dict,
    start_time: float
) -> None:
    """Writes metadata description JSON file alongside the downloaded files.
    
    Args:
        output_path (Path): Filepath to save metadata (.json).
        city (str): City name.
        year (int): Calendar target year.
        image_count (int): Count of images processed.
        qc_metrics (dict): Run metrics from GEE.
        start_time (float): Start execution epoch.
    """
    metadata = {
        "city": city,
        "year": year,
        "collection": Config.SATELLITE_COLLECTION,
        "cloud_probability_collection": Config.SATELLITE_CLOUD_PROBABILITY_COLLECTION,
        "cloud_threshold_pct": Config.SATELLITE_CLOUD_PERCENTAGE,
        "cloud_probability_threshold": Config.SATELLITE_CLOUD_PROBABILITY_THRESHOLD,
        "bands_extracted": Config.SATELLITE_BANDS,
        "projection_crs": qc_metrics.get("crs", Config.SATELLITE_EXPORT_CRS),
        "target_resolution_m": Config.SATELLITE_SCALE,
        "source_scene_count": image_count,
        "quality_metrics": {
            "qc_passed": qc_metrics.get("passed", False),
            "cloud_free_pixel_ratio": qc_metrics.get("valid_pixel_ratio", 0.0),
            "approx_total_pixels": qc_metrics.get("total_pixels", 0),
            "approx_valid_pixels": qc_metrics.get("valid_pixels", 0)
        },
        "pipeline_version": PIPELINE_VERSION,
        "git_commit": get_git_commit(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execution_stats": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_duration_sec": round(time.time() - start_time, 2)
        }
    }
    
    try:
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.success(f"Metadata parameters saved successfully at: {output_path}")
    except Exception as e:
        logger.error(f"Failed to write metadata JSON file: {e}")

def download_locally(
    image: ee.Image, 
    region: ee.Geometry, 
    output_tif_path: Path,
    scale: int = None
) -> bool:
    """Downloads a clipped multi-band image composite directly as a GeoTIFF.
    
    Requests download zip URL from Earth Engine API, downloads payload, extracts the 
    zip archive, and moves the GeoTIFF into the target folder.
    
    Args:
        image (ee.Image): GEE Image composite.
        region (ee.Geometry): Ingestion boundaries.
        output_tif_path (Path): Filepath to write output GeoTIFF.
        scale (int, optional): Download scale resolution. Defaults to Config.SATELLITE_SCALE.
        
    Returns:
        bool: True if download completed successfully, False otherwise.
    """
    dl_scale = scale if scale else Config.SATELLITE_SCALE
    logger.info(f"Initiating direct local download (scale={dl_scale}m)...")
    
    try:
        # Configure raw download details
        dl_params = {
            "name": "sentinel_image",
            "scale": dl_scale,
            "crs": Config.SATELLITE_EXPORT_CRS,
            "region": region,
            "format": "GEO_TIFF"
        }
        
        url = image.getDownloadURL(dl_params)
        logger.debug(f"Acquired GEE download link. Downloading content...")
        
        response = requests_get_with_retry(url, stream=True, timeout=120)
        if response.status_code != 200:
            logger.error(
                f"Direct download request rejected by GEE. Code: {response.status_code}\n"
                f"Detail: {response.text}"
            )
            return False
            
        # Parse returned zip contents in-memory
        zip_bytes = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_bytes) as z:
            file_list = z.namelist()
            tif_files = [f for f in file_list if f.endswith(".tif")]
            
            if not tif_files:
                logger.error("No GeoTIFF found inside the downloaded GEE zip archive.")
                return False
                
            # Sentinel-2 raw download may write bands as single or separate files.
            # Usually multi-band images download as "sentinel_image.tif" or separate bands if partitioned.
            # We extract and write the primary tif.
            target_name = tif_files[0]
            logger.info(f"Extracting {target_name} from archive...")
            
            # Write extracted content directly to output path
            output_tif_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_tif_path, "wb") as f:
                f.write(z.read(target_name))
                
        logger.success(f"GeoTIFF downloaded and saved successfully at: {output_tif_path}")
        return True
        
    except Exception as e:
        logger.error(
            f"Local direct download failed: {e}\n"
            "This can occur if the region area exceeds GEE's 32MB payload limit. "
            "Try increasing the scale (e.g. 50 or 100) or check configuration CRS."
        )
        return False

def export_to_drive(
    image: ee.Image, 
    region: ee.Geometry, 
    description: str, 
    folder: str,
    scale: int = None
) -> str:
    """Creates an asynchronous Earth Engine export task saving the image composite to Google Drive.
    
    Highly recommended for large, high-resolution (10m) city-wide composites.
    
    Args:
        image (ee.Image): GEE Image composite.
        region (ee.Geometry): Study boundary coordinates.
        description (str): Task description label.
        folder (str): Google Drive destination folder name.
        scale (int, optional): Export resolution. Defaults to Config.SATELLITE_SCALE.
        
    Returns:
        str: Task ID allocated by Earth Engine.
    """
    dl_scale = scale if scale else Config.SATELLITE_SCALE
    logger.info(f"Creating asynchronous batch GEE export task to Drive folder '{folder}'...")
    
    try:
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            fileNamePrefix=description,
            scale=dl_scale,
            crs=Config.SATELLITE_EXPORT_CRS,
            region=region,
            maxPixels=1e10
        )
        task.start()
        
        task_id = task.id
        logger.success(f"Asynchronous GEE export task created successfully. Task ID: {task_id}")
        logger.info("Check status via command line: 'earthengine task list'")
        return task_id
        
    except Exception as e:
        logger.error(f"Failed to submit batch GEE Drive export task: {e}")
        return ""
