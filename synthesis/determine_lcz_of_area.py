import json
from pathlib import Path
from synthesis.determine_lcz_parameters import get_ndvi_features, get_building_features, get_building_footprints
import numpy as np
import re
import geopandas as gpd
from shapely.geometry import shape
import logging
import os

def determine_lcz_of_area(polygon: str | Path | dict):

    logging.info(f"Determining urban class of area")

    area_type = {'dense': "dense_built_up",
                 'mid': 'mid_built_up',
                 'low': 'low_built_up'}

    if polygon is None:
        print("No input polygon provided")
        return
    _path = "DATA/OUT/temp_geometry.gpkg"

    if type(polygon) == dict:
        logging.info(polygon)
        _polygon = shape(polygon["geometry"])
        gdf = gpd.GeoDataFrame(
            [{"geometry": _polygon}],
            crs="EPSG:4326"
        )
        gdf.to_file(_path, driver="GPKG", layer="aoi")
        polygon = _path
    
    name = str(polygon)


    train_parameters = load_lcz_parameters()
    area_parameters = determine_lcz_parameters(polygon)

    ranking = compare_lcz_parameters(area_parameters, train_parameters)

    for class_name, distance in ranking:
        print(f"Class: {class_name}, Distance: {distance:.4f}")

    print(f"Most likely LCZ class for area '{name}': {ranking[0][0]}")


    if os.path.exists(_path):
        os.remove(_path)

    return area_type[ranking[0][0]]

def compare_lcz_parameters(area_parameters, train_parameters):
    """Rank LCZ classes by standardized Euclidean distance."""

    # Calculate a representative profile for each class
    profiles = group_training_parameters(train_parameters)

    # Features available in both the new area and class profiles
    features = [
        feature
        for feature, value in area_parameters.items()
        if isinstance(value, (int, float))
        and np.isfinite(value)
        and all(feature in profile for profile in profiles.values())
    ]

    if not features:
        raise ValueError("No common numeric features to compare.")

    # Standardization parameters from all individual training areas
    training_values = {
        feature: [
            area[feature]
            for area in train_parameters.values()
            if feature in area
            and isinstance(area[feature], (int, float))
            and np.isfinite(area[feature])
        ]
        for feature in features
    }

    means = {
        feature: np.mean(values)
        for feature, values in training_values.items()
    }
    stds = {
        feature: np.std(values)
        for feature, values in training_values.items()
    }

    # Avoid division by zero for constant features
    stds = {
        feature: std if std > 0 else 1.0
        for feature, std in stds.items()
    }

    distances = {}

    for class_name, profile in profiles.items():
        squared_differences = []

        for feature in features:
            area_z = (
                area_parameters[feature] - means[feature]
            ) / stds[feature]

            class_z = (
                profile[feature] - means[feature]
            ) / stds[feature]

            squared_differences.append((area_z - class_z) ** 2)

        distances[class_name] = float(
            np.sqrt(np.mean(squared_differences))
        )

    return sorted(distances.items(), key=lambda item: item[1])

def group_training_parameters(train_parameters):
    """Group training areas by class and calculate mean feature values."""
    grouped = {}

    for area_name, features in train_parameters.items():
        match = re.search(r"_(dense|mid|low)-", area_name)
        if not match:
            continue

        class_name = match.group(1)
        grouped.setdefault(class_name, []).append(features)

    class_profiles = {}

    for class_name, areas in grouped.items():
        class_profiles[class_name] = {}

        # Only average numeric features available in the training data
        for feature in areas[0]:
            values = [
                area[feature]
                for area in areas
                if feature in area
                and area[feature] is not None
                and isinstance(area[feature], (int, float))
                and np.isfinite(area[feature])
            ]

            if values:
                class_profiles[class_name][feature] = float(
                    np.mean(values)
                )

    return class_profiles

def load_lcz_parameters():
    lcz_features_path = Path("synthesis/tuning-files/ndvi_building_features.json")

    with open(lcz_features_path, "r") as f:
            lcz_features = json.load(f)

    return lcz_features

def determine_lcz_parameters(polygon: str | Path):
    ndvi_features = get_ndvi_features(polygon)
    buildings = get_building_footprints(polygon)
    building_features = get_building_features(buildings = buildings, area = polygon)

    area_features = {**ndvi_features, **building_features}

    return area_features

def main():
    poly = {'type': 'Feature', 'properties': {}, 'geometry': {'type': 'Polygon', 'coordinates': [[[4.472744, 51.926352], [4.472744, 51.929567], [4.480319, 51.929567], [4.480319, 51.926352], [4.472744, 51.926352]]]}}
    print(determine_lcz_of_area(poly))

if __name__ == "__main__":
    main()