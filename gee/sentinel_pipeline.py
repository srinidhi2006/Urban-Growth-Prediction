"""
Sentinel-2 Composite Building and Quality Check Pipeline.
Applies temporal, spatial, and cloud-masking filters inside Google Earth Engine.
"""

import ee
from loguru import logger
from config import Config

def mask_s2_clouds_advanced(image: ee.Image) -> ee.Image:
    """Masks clouds in Sentinel-2 imagery using QA60 bitmask AND S2 Cloud Probability.
    
    Args:
        image (ee.Image): Sentinel-2 input image with joined 'cloud_mask' property.
        
    Returns:
        ee.Image: Cloud-masked Sentinel-2 image.
    """
    # 1. QA60 Masking
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    qa_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
              qa.bitwiseAnd(cirrus_bit_mask).eq(0))
              
    # 2. Cloud Probability Masking
    has_prob = image.propertyNames().contains("cloud_mask")
    
    # Retrieve probability and create threshold mask
    cloud_prob_img = ee.Image(image.get("cloud_mask"))
    prob = cloud_prob_img.select("probability")
    prob_mask = prob.lt(Config.SATELLITE_CLOUD_PROBABILITY_THRESHOLD)
    
    # Merge masks
    combined_mask = qa_mask.And(prob_mask)
    
    # Return masked image based on property availability
    final_mask = ee.Image(ee.Algorithms.If(has_prob, combined_mask, qa_mask))
    
    return image.updateMask(final_mask)

def execute_quality_check(image: ee.Image, boundary: ee.Geometry) -> dict:
    """Verifies that the compiled image composite meets quality requirements.
    
    Validates bands presence, resolution matches, CRS definitions, and valid (unmasked)
    pixel ratio inside the target boundary.
    
    Args:
        image (ee.Image): Clipped image composite to inspect.
        boundary (ee.Geometry): Administrative study boundary.
        
    Returns:
        dict: Diagnostic metrics including quality check execution status.
    """
    logger.info("Executing Quality Check validation on GEE image composite...")
    try:
        # 1. Band verification
        band_names = image.bandNames().getInfo()
        missing_bands = [b for b in Config.QC_EXPECTED_BANDS if b not in band_names]
        
        # 2. CRS & Scale validation (Use the first band, e.g. B2)
        proj = image.select(Config.QC_EXPECTED_BANDS[0]).projection()
        crs = proj.crs().getInfo()
        nominal_scale = proj.nominalScale().getInfo()
        
        # 3. Cloud-free Pixel Density Ratio
        # Create a constant image = 1, clipped to boundary (acts as boundary mask helper)
        constant_img = ee.Image.constant(1).clip(boundary)
        # Apply the mask of the processed image to check for empty/cloud-masked voids
        masked_constant = constant_img.updateMask(image.select(Config.QC_EXPECTED_BANDS[0]).mask())
        
        # Calculate pixel counts inside boundary. We use a coarser resolution (e.g. 100m)
        # for rapid server-side calculations to avoid execution timeouts.
        counts = ee.Image.cat([constant_img.rename("total"), masked_constant.rename("valid")]).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=boundary,
            scale=100,
            maxPixels=1e8
        ).getInfo()
        
        total_pixels = counts.get("total", 0)
        valid_pixels = counts.get("valid", 0)
        
        valid_ratio = (valid_pixels / total_pixels) if total_pixels > 0 else 0.0
        
        # Evaluate strict conditions
        bands_ok = len(missing_bands) == 0
        scale_ok = nominal_scale <= Config.QC_EXPECTED_RESOLUTION * 2.0
        ratio_ok = valid_ratio >= Config.QC_MIN_CLOUD_FREE_RATIO
        
        passed = bool(bands_ok and scale_ok and ratio_ok)
        
        metrics = {
            "passed": passed,
            "crs": crs,
            "resolution": nominal_scale,
            "valid_pixel_ratio": round(valid_ratio, 4),
            "missing_bands": missing_bands,
            "total_pixels": total_pixels,
            "valid_pixels": valid_pixels
        }
        
        if passed:
            logger.success(
                f"Quality Check PASSED: Valid Pixel Ratio = {metrics['valid_pixel_ratio']:.2%}, "
                f"CRS = {crs}, Resolution = {nominal_scale}m."
            )
        else:
            logger.warning(
                f"Quality Check FAILED: Valid Pixel Ratio = {metrics['valid_pixel_ratio']:.2%} "
                f"(min: {Config.QC_MIN_CLOUD_FREE_RATIO:.2%}), Missing bands: {missing_bands}."
            )
            
        return metrics
        
    except Exception as e:
        logger.error(f"Error executing composite quality check: {e}")
        return {
            "passed": False,
            "error": str(e),
            "valid_pixel_ratio": 0.0,
            "missing_bands": Config.QC_EXPECTED_BANDS
        }

def build_sentinel_composite(
    city_name: str, 
    year: int, 
    boundary: ee.Geometry,
    start_date: str = None,
    end_date: str = None
) -> tuple:
    """Queries Sentinel-2 surface reflectance, filters, cloud-masks, and builds a median composite.
    
    Args:
        city_name (str): Target city.
        year (int): Year for imagery (e.g. 2019, 2026).
        boundary (ee.Geometry): Geospatial boundary to clip.
        start_date (str, optional): Start date string (YYYY-MM-DD). Defaults to year-01-01.
        end_date (str, optional): End date string (YYYY-MM-DD). Defaults to year-12-31.
        
    Returns:
        tuple: (ee.Image composite, dict QC metrics, int image count in initial filtered collection)
        
    Raises:
        ValueError: If the initial query returns an empty collection.
    """
    logger.info(f"Querying Sentinel-2 collection for {city_name} in {year}...")
    
    # Resolve dates
    query_start = start_date if start_date else f"{year}-01-01"
    query_end = end_date if end_date else f"{year}-12-31"
    
    logger.info(f"Temporal range: {query_start} to {query_end}")
    logger.info(f"Max cloud coverage filter: {Config.SATELLITE_CLOUD_PERCENTAGE}%")
    
    # 1. Query S2 Surface Reflectance
    s2_sr = (
        ee.ImageCollection(Config.SATELLITE_COLLECTION)
        .filterBounds(boundary)
        .filterDate(query_start, query_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", Config.SATELLITE_CLOUD_PERCENTAGE))
    )
    
    # 2. Query matching S2 Cloud Probability
    s2_cloud_prob = (
        ee.ImageCollection(Config.SATELLITE_CLOUD_PROBABILITY_COLLECTION)
        .filterBounds(boundary)
        .filterDate(query_start, query_end)
    )
    
    # 3. Inner join the two collections by their index
    join = ee.Join.saveFirst(
        matchKey="cloud_mask",
        ordering="system:time_start",
        ascending=True
    )
    filter_by_id = ee.Filter.equals(
        leftField="system:index",
        rightField="system:index"
    )
    joined_collection = join.apply(s2_sr, s2_cloud_prob, filter_by_id)
    
    # Get collection size
    image_count = int(joined_collection.size().getInfo())
    logger.info(f"Retrieved {image_count} scenes matching filtering criteria.")
    
    if image_count == 0:
        err_msg = f"Zero Sentinel-2 images found for {city_name} between {query_start} and {query_end}."
        logger.critical(err_msg)
        raise ValueError(err_msg)
        
    # 4. Preprocess: Mask clouds and create Median Composite
    composite = joined_collection.map(mask_s2_clouds_advanced).median()
    
    # 5. Select target bands and clip to boundaries
    selected_composite = composite.select(Config.SATELLITE_BANDS).clip(boundary)
    
    # 6. Quality Check
    qc_metrics = execute_quality_check(selected_composite, boundary)
    
    return selected_composite, qc_metrics, image_count
