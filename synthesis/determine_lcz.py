import geopandas as gpd
import ee
from numpy.compat import Path
import pystac_client
import planetary_computer
import rioxarray
import numpy as np
import matplotlib.pyplot as plt

def plot_rgb(red, green, blue, title="RGB Image"):
    rgb = np.stack([red.values, green.values, blue.values], axis=-1)

    # Sentinel-2 reflectance needs scaling
    rgb = rgb / 10000

    # Improve visualisation
    rgb = np.clip(rgb * 3, 0, 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

def plot_ndvi(ndvi, title="NDVI"):
    plt.figure(figsize=(10, 10))
    plt.imshow(ndvi.values, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(label="NDVI")
    plt.title(title)
    plt.axis("off")
    plt.show()

def get_ndvi_features(gpkg_path: str | Path, plot_rgb_bool: bool = False, plot_ndvi_bool: bool = False):
    gdf = gpd.read_file(gpkg_path)

    # Convert to WGS84 (important for many web/satellite APIs)
    gdf = gdf.to_crs("EPSG:4326")

    polygon = gdf.geometry.iloc[0]

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1/",
        modifier=planetary_computer.sign_inplace
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=polygon.__geo_interface__,
        datetime="2026-05-01/2026-09-01",
        query={"eo:cloud_cover": {"lt": 20}}
    )

    items = list(search.items())

    print(f"Found {len(items)} images")

    item = min(
        items,
        key=lambda x: x.properties["eo:cloud_cover"]
    )

    print(item.id)
    print(item.properties["eo:cloud_cover"])

    #Extract the URLs for the bands needed
    green_url = item.assets["B03"].href
    blue_url = item.assets["B02"].href
    red_url = item.assets["B04"].href
    nir_url = item.assets["B08"].href



    red = rioxarray.open_rasterio(red_url, masked=True).squeeze()
    green = rioxarray.open_rasterio(green_url, masked=True).squeeze()
    blue = rioxarray.open_rasterio(blue_url, masked=True).squeeze()
    nir = rioxarray.open_rasterio(nir_url, masked=True).squeeze()

    # Clip to polygon
    polygon_raster = gdf.to_crs(red.rio.crs)

    red = red.rio.clip(polygon_raster.geometry)
    green = green.rio.clip(polygon_raster.geometry)
    blue = blue.rio.clip(polygon_raster.geometry)
    nir = nir.rio.clip(polygon_raster.geometry)

    ndvi = (nir - red) / (nir + red)

    ndvi = ndvi.squeeze()

    values = ndvi.values.flatten()
    values = values[~np.isnan(values)]

    features = {
        "ndvi_mean": np.mean(values),
        "ndvi_median": np.median(values),
        "ndvi_std": np.std(values),
        "ndvi_min": np.min(values),
        "ndvi_max": np.max(values),
        "ndvi_p25": np.percentile(values, 25),
        "ndvi_p75": np.percentile(values, 75),
        "vegetation_fraction": np.mean(values > 0.3)
    }

    print(features)
    if plot_rgb_bool:
        plot_rgb(red, green, blue, title="Sentinel-2 True Colour")
    if plot_ndvi_bool:
        plot_ndvi(ndvi, title="NDVI")

def main():
    ndvi_features = get_ndvi_features("data/in/delft_dense-1/delft-dense-1.gpkg", plot_rgb_bool=True, plot_ndvi_bool=True)


if __name__ == "__main__":
    main()