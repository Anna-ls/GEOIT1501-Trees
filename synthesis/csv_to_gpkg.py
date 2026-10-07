import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Define your file paths
input_csv = "C:/Users/annas/Documents/GEOIT1501-Trees/DATA/inventory/barcelona_full_trees.csv"
output_gpkg = "C:/Users/annas/Documents/GEOIT1501-Trees/DATA/inventory/barcelona_full_trees.gpkg"

# 1. Read the CSV into a pandas DataFrame
df = pd.read_csv(input_csv)

# 2. Create geometry column from the lat/lon columns
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]

# 3. Convert the pandas DataFrame to a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# 4. Save the GeoDataFrame to a GeoPackage
gdf.to_file(output_gpkg, driver="GPKG")

print(f"Successfully created {output_gpkg}!")