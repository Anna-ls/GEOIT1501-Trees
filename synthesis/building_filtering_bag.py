import geopandas as gpd
from pathlib import Path
import requests
from shapely.geometry import shape
from shapely import contains_xy
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree



def get_building_footprints(polygon, crs="EPSG:28992"):
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

    print(
        f"  [+] Querying BAG buildings inside "
        f"{minx:.1f}, {miny:.1f}, "
        f"{maxx:.1f}, {maxy:.1f}"
    )

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
        "limit": 10000,
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

        response.raise_for_status()
        data = response.json()
        page_features = data.get("features", [])
        features.extend(page_features)

        print(
            f"      Retrieved {len(page_features):,} "
            f"features "
            f"(total: {len(features):,})"
        )

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

    print(
        f"  [✓] Found {len(buildings):,} "
        f"BAG building footprints."
    )

    return buildings

def filter_trees_in_buildings_bag(labels, pts_xy, translation, epsg, buffer=0.5):
    """Zero out labels of trees whose centroid sits inside a buffered BAG footprint."""
    if epsg is None:
        return labels, 0

    unique = np.unique(labels[labels > 0])
    if len(unique) == 0:
        return labels, 0

    tx, ty = float(translation[0]), float(translation[1])

    # Calculate per-tree centroids in projected coords
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    sum_x = np.bincount(labels, weights=pts_xy[:, 0], minlength=counts.size)
    sum_y = np.bincount(labels, weights=pts_xy[:, 1], minlength=counts.size)
    centroids = np.column_stack(
        [sum_x[unique] / counts[unique] + tx, sum_y[unique] / counts[unique] + ty]
    )

    # Generate a bounding box polygon for the BAG query
    minx, miny = pts_xy[:, 0].min() + tx, pts_xy[:, 1].min() + ty
    maxx, maxy = pts_xy[:, 0].max() + tx, pts_xy[:, 1].max() + ty
    bbox_poly = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])

    # Fetch BAG footprints
    try:
        buildings = get_building_footprints(bbox_poly, crs=f"EPSG:{epsg}")
    except Exception as e:
        print(f"    buildings (BAG): API failed ({e}), no trees removed.")
        return labels, 0

    if buildings.empty:
        return labels, 0

    # Filter tree centroids against building polygons
    geoms = [geom.buffer(buffer) if buffer else geom for geom in buildings.geometry]
    tree = STRtree(geoms)

    inside = np.zeros(len(centroids), dtype=bool)
    for i, xy in enumerate(centroids):
        if len(tree.query(Point(xy), predicate="intersects")) > 0:
            inside[i] = True

    removed = unique[inside]
    if len(removed):
        labels[np.isin(labels, removed)] = 0

    return labels, int(len(removed))

if __name__ == "__main__":

    buildings = get_building_footprints(
        "DATA/IN/delft_dense-1/delft-dense-1.gpkg",
        crs="EPSG:28992"
    )

    buildings.to_file(
        "DATA/IN/delft_dense-1/delft-dense-1-buildings.gpkg",
        driver="GPKG"
    )
