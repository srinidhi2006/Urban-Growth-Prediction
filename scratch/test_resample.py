import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from pathlib import Path

def main():
    project_root = Path(".").resolve()
    grid_path = project_root / "data" / "interim" / "osm" / "Pune" / "grid.geojson"
    raster_path = project_root / "data" / "processed" / "sentinel" / "Pune" / "2026" / "Pune_2026_mosaic.tif"
    
    grid = gpd.read_file(grid_path)
    src = rasterio.open(raster_path)
    
    # Project grid to match raster CRS
    grid = grid.to_crs(src.crs)
    
    # Try crop first cell
    geom = grid.geometry.iloc[0]
    out_image, out_transform = mask(src, [geom], crop=True)
    
    print("Original crop shape:", out_image.shape)
    
    # Resample using MemoryFile
    with MemoryFile() as memfile:
        profile = src.profile.copy()
        profile.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
        })
        with memfile.open(**profile) as mem_dst:
            mem_dst.write(out_image)
            resized = mem_dst.read(
                out_shape=(src.count, 128, 128),
                resampling=Resampling.bilinear
            )
            
    print("Resampled shape:", resized.shape)

if __name__ == "__main__":
    main()
