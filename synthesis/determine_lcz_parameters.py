import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import geopandas as gpd
import pystac_client
import planetary_computer
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import requests
from shapely.geometry import shape
import math
import json

def json_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )

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

def get_ndvi_features(gpkg_path: str | Path, plot_rgb_bool: bool = False, plot_ndvi_bool: bool = False, name: str = "default"):
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

    # print(f"Found {len(items)} images")

    item = min(
        items,
        key=lambda x: x.properties["eo:cloud_cover"]
    )

    # print(f"Item ID: {item.id}")
    # print(f"Cloud Cover: {item.properties['eo:cloud_cover']}")

    # print(f"Extracting bands for NDVI calculation...")
    #Extract the URLs for the bands needed
    green_url = item.assets["B03"].href
    blue_url = item.assets["B02"].href
    red_url = item.assets["B04"].href
    nir_url = item.assets["B08"].href


    if plot_rgb_bool:
        green = rioxarray.open_rasterio(green_url, masked=True).squeeze()
        blue = rioxarray.open_rasterio(blue_url, masked=True).squeeze()

    red = rioxarray.open_rasterio(red_url, masked=True).squeeze()
    nir = rioxarray.open_rasterio(nir_url, masked=True).squeeze()

    # Clip to polygon
    polygon_raster = gdf.to_crs(red.rio.crs)


    if plot_rgb_bool:
        green = green.rio.clip(polygon_raster.geometry)
        blue = blue.rio.clip(polygon_raster.geometry)


    # Get the polygon bounding box
    minx, miny, maxx, maxy = polygon_raster.total_bounds

    # Crop to the bounding box first, then mask to the exact polygon
    red = red.rio.clip_box(
        minx=minx, miny=miny, maxx=maxx, maxy=maxy
    ).rio.clip(polygon_raster.geometry, drop=True)

    nir = nir.rio.clip_box(
        minx=minx, miny=miny, maxx=maxx, maxy=maxy
    ).rio.clip(polygon_raster.geometry, drop=True)


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


    if plot_rgb_bool:
        plot_rgb(red, green, blue, title="Sentinel-2 True Colour")
    if plot_ndvi_bool:
        plot_ndvi(ndvi, title="NDVI")

    return features

def plot_results(
    input: dict,
    output_path: str = "features_plot.png",
    class_order: list = ("dense", "mid", "low"),
    title: str = "Feature Comparison by Urban Class",
    features: list = None,
):
    # -----------------------------
    # 1. Prepare the data
    # -----------------------------
    df = pd.DataFrame.from_dict(input, orient="index")
    df.index.name = "Area"
    df = df.reset_index()

    if "class" not in df.columns:
        df["class"] = df["Area"].str.extract(
            r"_(dense|low|mid)-"
        )

    if "city" not in df.columns:
        df["city"] = df["Area"].str.extract(
            r"^(delft|utrecht|amsterdam)_"
        )

    class_order = list(class_order)

    # Consistent colours across every feature
    palette = {
        "dense": "#F7C027",  # Blue
        "mid":   "#BCEC20",  # Teal
        "low":   "#0DBF0A",  # Warm orange
    }

    # Allow custom class names without causing errors
    for cls in class_order:
        if cls not in palette:
            palette[cls] = "#999999"

    # Automatically select numerical features
    if features is None:
        exclude = {"Area", "class", "city"}
        features = [
            col for col in df.select_dtypes(include="number").columns
            if col not in exclude
        ]
    else:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Features not found: {missing}")

        non_numeric = [
            f for f in features
            if not pd.api.types.is_numeric_dtype(df[f])
        ]
        if non_numeric:
            raise ValueError(
                f"Features must be numerical: {non_numeric}"
            )

    if not features:
        raise ValueError("No numerical features found to plot.")

    # -----------------------------
    # 2. Set a clean visual style
    # -----------------------------
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font="DejaVu Sans",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#333333",
            "text.color": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "grid.color": "#D9DEE5",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
        },
    )

    # Flexible grid layout
    ncols = min(3, len(features))
    nrows = math.ceil(len(features) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.2 * nrows),
        squeeze=False,
    )
    axes = axes.flatten()

    # -----------------------------
    # 3. Plot each feature
    # -----------------------------
    for ax, feature in zip(axes, features):
        sns.boxplot(
            data=df,
            x="class",
            y=feature,
            order=class_order,
            hue="class",
            hue_order=class_order,
            palette=palette,
            dodge=False,
            legend=False,
            showfliers=False,
            width=0.55,
            linewidth=1.3,
            saturation=0.9,
            boxprops={"edgecolor": "#FFFFFF"},
            whiskerprops={"color": "#555555", "linewidth": 1.2},
            capprops={"color": "#555555", "linewidth": 1.2},
            medianprops={"color": "#222222", "linewidth": 2},
            ax=ax,
        )

        # Overlay individual observations
        sns.stripplot(
            data=df,
            x="class",
            y=feature,
            order=class_order,
            hue="class",
            hue_order=class_order,
            palette=palette,
            dodge=False,
            legend=False,
            jitter=0.14,
            size=4,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.4,
            ax=ax,
        )

        # Titles and labels
        ax.set_title(
            feature.replace("_", " ").title(),
            fontsize=12,
            fontweight="semibold",
            loc="left",
            pad=12,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.tick_params(axis="x", labelsize=10, length=0)
        ax.tick_params(axis="y", labelsize=9)

        ax.grid(axis="y", linestyle="-", alpha=0.65)
        ax.grid(axis="x", visible=False)
        ax.set_facecolor("#FFFFFF")

    # Remove unused axes
    for ax in axes[len(features):]:
        fig.delaxes(ax)

    # -----------------------------
    # 4. Final layout and export
    # -----------------------------
    fig.suptitle(
        title,
        fontsize=17,
        fontweight="bold",
        x=0.06,
        ha="left",
        y=1.01,
    )

    fig.patch.set_facecolor("#FFFFFF")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()
    plt.close(fig)

    return df


def get_building_footprints(polygon, crs="EPSG:28992", plot_buildings=False):
    """
    Retrieve all BAG building footprints intersecting a polygon.

    Parameters
    ----------
    polygon : str, Path, GeoDataFrame, or Shapely geometry
        Input polygon. Can be:
        - path to a GeoPackage
        - GeoDataFrame
        - Shapely Polygon/MultiPolygon

    crs : str
        Target CRS. Default: EPSG:28992.

    Returns
    -------
    GeoDataFrame
        BAG building footprints intersecting the polygon.
    """

    # ---------------------------------------------------------------
    # 1. Load/prepare polygon
    # ---------------------------------------------------------------

    if isinstance(polygon, (str, Path)):

        polygon_gdf = gpd.read_file(polygon)

        if polygon_gdf.empty:
            raise ValueError(
                f"No geometries found in {polygon}"
            )

        if polygon_gdf.crs is None:
            raise ValueError(
                "Input polygon has no CRS."
            )

        polygon_gdf = polygon_gdf.to_crs(crs)

    elif isinstance(polygon, gpd.GeoDataFrame):

        polygon_gdf = polygon.copy()

        if polygon_gdf.empty:
            raise ValueError(
                "Input GeoDataFrame contains no geometries."
            )

        if polygon_gdf.crs is None:
            raise ValueError(
                "Input GeoDataFrame has no CRS."
            )

        polygon_gdf = polygon_gdf.to_crs(crs)

    else:

        # Shapely Polygon / MultiPolygon
        polygon_gdf = gpd.GeoDataFrame(
            {"geometry": [polygon]},
            geometry="geometry",
            crs=crs
        )

    # Dissolve multiple features into one geometry
    polygon_geom = polygon_gdf.geometry.union_all()

    minx, miny, maxx, maxy = polygon_geom.bounds

    # print(
    #     f"  [+] Querying BAG buildings inside "
    #     f"{minx:.1f}, {miny:.1f}, "
    #     f"{maxx:.1f}, {maxy:.1f}"
    # )

    # ---------------------------------------------------------------
    # 2. BAG OGC API
    # ---------------------------------------------------------------

    url = (
        "https://api.pdok.nl/kadaster/bag/ogc/v2/"
        "collections/pand/items"
    )

    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bbox-crs": (
            "http://www.opengis.net/def/crs/EPSG/0/28992"
        ),
        "crs": (
            "http://www.opengis.net/def/crs/EPSG/0/28992"
        ),
        "f": "json",
        "limit": 2000,
    }

    features = []

    # ---------------------------------------------------------------
    # 3. Download all pages
    # ---------------------------------------------------------------

    while url:

        response = requests.get(
            url,
            params=params,
            timeout=120
        )
        try:
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error occurred while fetching data from {url}: {e}")
            continue

        data = response.json()

        page_features = data.get("features", [])

        features.extend(page_features)

        # print(
        #     f"      Retrieved {len(page_features):,} "
        #     f"features "
        #     f"(total: {len(features):,})"
        # )

        # Find next page
        next_url = None

        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_url = link.get("href")
                break

        url = next_url

        # The next URL already contains its parameters
        params = None

    # ---------------------------------------------------------------
    # 4. Convert GeoJSON → GeoDataFrame
    # ---------------------------------------------------------------

    if not features:

        print("  [!] No BAG buildings returned.")

        return gpd.GeoDataFrame(
            geometry=[],
            crs=crs
        )

    geometries = []
    properties = []

    for feature in features:

        geometry = feature.get("geometry")

        if geometry is None:
            continue

        geometries.append(
            shape(geometry)
        )

        properties.append(
            feature.get("properties", {})
        )

    buildings = gpd.GeoDataFrame(
        properties,
        geometry=geometries,
        crs=crs
    )

    # ---------------------------------------------------------------
    # 5. Exact spatial filtering
    # ---------------------------------------------------------------

    buildings = buildings[
        buildings.geometry.intersects(polygon_geom)
    ].copy()

    buildings.reset_index(drop=True, inplace=True)

    # print(
    #     f"  [✓] Found {len(buildings):,} "
    #     f"BAG building footprints."
    # )

    

    if plot_buildings:
        # Plot building footprints
        fig, ax = plt.subplots(figsize=(10, 10))
        buildings.plot(
            ax=ax,
            facecolor="lightblue",
            edgecolor="black",
            linewidth=0.3
        )

        # Plot the study area boundary
        study_area = gpd.GeoSeries([polygon_geom], crs=buildings.crs)
        study_area.boundary.plot(
            ax=ax,
            color="red",
            linewidth=2
        )

        ax.set_title("Building footprints within study area")
        ax.set_aspect("equal")
        ax.set_axis_off()

        plt.show()

    return buildings

def get_building_features(buildings: gpd.GeoDataFrame, area: str | Path, crs="EPSG:28992") -> dict:
    area_gdf = gpd.read_file(area)
    area_gdf = area_gdf.set_crs(crs, allow_override=True) if area_gdf.crs is None else area_gdf

    # Reproject to a suitable projected CRS in metres
    metric_crs = buildings.estimate_utm_crs()

    buildings_m = buildings.to_crs(metric_crs)
    study_area = area_gdf.geometry.union_all()

    # Reproject to the metric CRS
    study_area = (
        gpd.GeoSeries([study_area], crs=area_gdf.crs)
        .to_crs(metric_crs)
        .iloc[0]
    )

    area_m2 = study_area.area

    # Keep buildings intersecting the study area
    intersecting = buildings_m[
        buildings_m.geometry.intersects(study_area)
    ].copy()

    # Clip footprints to the study area
    clipped = intersecting.geometry.intersection(study_area)
    clipped = clipped[
        (~clipped.is_empty) & (clipped.area > 0)
    ]

    if area_m2 <= 0:
        raise ValueError("Study area must have positive area.")

    if len(clipped) == 0:
        return {
            "study_area_m2": area_m2,
            "building_coverage_ratio": 0.0,
            "building_count": 0,
            "buildings_per_hectare": 0.0,
        }

    # Union prevents overlapping footprints being counted twice
    building_union = clipped.union_all()
    total_built_area = building_union.area

    # Use individual clipped footprints for basic statistics
    areas = clipped.area
    perimeters = clipped.length

    # Compactness: 1 for a perfect circle, lower for less compact shapes
    compactness = (
        4 * np.pi * areas / (perimeters ** 2)
    ).replace([np.inf, -np.inf], np.nan)

    features = {
        "study_area_m2": area_m2,
        "building_covered_area_m2": total_built_area,
        "building_coverage_ratio": (
            total_built_area / area_m2
        ),
        "building_coverage_percent": (
            100 * total_built_area / area_m2
        ),
        "building_count": len(clipped),
        "buildings_per_hectare": (
            len(clipped) / (area_m2 / 10000)
        ),
        "mean_building_area_m2": areas.mean(),
        "median_building_area_m2": areas.median(),
        "std_building_area_m2": areas.std(),
        "p90_building_area_m2": areas.quantile(0.9),
        "mean_building_compactness": compactness.mean(),
    }

    return features

def main():
    train_files = [
        #Dense urban areas
        "data/in/delft_dense-1/delft-dense-1.gpkg",
        "data/in/delft_dense-2/delft-dense-2.gpkg",
        "data/in/utrecht_dense-1/utrecht-dense-1.gpkg",
        "data/in/utrecht_dense-2/utrecht-dense-2.gpkg",
        "data/in/amsterdam_dense-1/amsterdam-dense-1.gpkg",
        "data/in/amsterdam_dense-2/amsterdam-dense-2.gpkg",

        #Low urban areas
        "data/in/delft_low-1/delft-low-1.gpkg",
        # "data/in/delft_low-2/delft-low-2.gpkg",
        # "data/in/utrecht_low-1/utrecht-low-1.gpkg",
        "data/in/utrecht_low-2/utrecht-low-2.gpkg",
        "data/in/amsterdam_low-1/amsterdam-low-1.gpkg",
        "data/in/amsterdam_low-2/amsterdam-low-2.gpkg",

        #Mid urban areas
        "data/in/delft_mid-1/delft-mid-1.gpkg",
        # "data/in/delft_mid-2/delft-mid-2.gpkg",
        "data/in/utrecht_mid-1/utrecht-mid-1.gpkg",
        "data/in/utrecht_mid-2/utrecht-mid-2.gpkg",
        "data/in/amsterdam_mid-1/amsterdam-mid-1.gpkg",
        "data/in/amsterdam_mid-2/amsterdam-mid-2.gpkg",

    ]

    results = {}

    """
        NDVI Features:
            "ndvi_mean"
            "ndvi_median"
            "ndvi_std"
            "ndvi_min"
            "ndvi_max"
            "ndvi_p25"
            "ndvi_p75"
            "vegetation_fraction"
        Building Features:
            "study_area_m2"
            "building_covered_area_m2"
            "building_coverage_ratio"
            "building_coverage_percent"
            "building_count"
            "buildings_per_hectare"
            "mean_building_area_m2"
            "median_building_area_m2"
            "std_building_area_m2"
            "p90_building_area_m2"
            "mean_building_compactness"
    """

    for train_file in tqdm(train_files):
        name = train_file.split("/")[-2]
        # print(f"Processing {name}...")
        ndvi_features = get_ndvi_features(train_file, plot_rgb_bool=False, plot_ndvi_bool=False, name=name)

        building_footprints = get_building_footprints(train_file, crs="EPSG:28992", plot_buildings=False)
        building_features = get_building_features(building_footprints, train_file, crs="EPSG:28992")
        results[name] = {**ndvi_features, **building_features}

    
    #Plot the results of the selected features
    plot_results(results, output_path="ndvi_features_plot.png", features=["building_coverage_ratio", 
                                                                          "ndvi_mean",
                                                                          "ndvi_median",
                                                                          ], title="Building and NDVIFeatures by Urban Class")

    with open("synthesis/tuning-files/ndvi_building_features.json", "w") as f:
        json.dump(results, f, indent=4, default=json_serializer)



if __name__ == "__main__":
    main()