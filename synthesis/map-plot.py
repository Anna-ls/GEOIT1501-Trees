import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
from pathlib import Path
import pandas as pd
from matplotlib.lines import Line2D

outlines_dir = Path("../data-2/in/outlines/amsterdam")
trees_dir = Path("../data-2/in/tree-inventories/amsterdam")
out_dir = Path("../data-2/out")
out_dir.mkdir(parents=True, exist_ok=True)

outline_files = sorted(outlines_dir.glob("*.gpkg"))
tree_files = sorted(trees_dir.glob("*.gpkg"))

ratio_width, ratio_height = 16, 9
fig, ax = plt.subplots(figsize=(ratio_width, ratio_height))
all_outlines = []

for i, outline_path in enumerate(outline_files):

    if i<2:
        edge_color = "#950606"
    elif i < 4:
        edge_color = "#198450"
    else:
        edge_color = "#ee7600"


    outline_wm = gpd.read_file(outline_path).dissolve().to_crs(epsg=3857)
    outline_wm.plot(ax=ax, facecolor="none", edgecolor=edge_color, linewidth=2, zorder=2)
    all_outlines.append(outline_wm)

combined_gdf = gpd.GeoDataFrame(pd.concat(all_outlines, ignore_index=True))
minx, miny, maxx, maxy = combined_gdf.total_bounds

center_x = (minx + maxx) / 2
center_y = (miny + maxy) / 2
data_width = maxx - minx
data_height = maxy - miny

target_ratio = ratio_width / ratio_height
buffer = 1000

if (data_width / data_height) > target_ratio:
    # Data is wider than 16:9
    total_x_span = data_width + (buffer * 2)
    total_y_span = total_x_span / target_ratio
else:
    # Data is taller than 16:9
    total_y_span = data_height + (buffer * 2)
    total_x_span = total_y_span * target_ratio

ax.set_xlim(center_x - (total_x_span / 2), center_x + (total_x_span / 2))
ax.set_ylim(center_y - (total_y_span / 2), center_y + (total_y_span / 2))

api_key = "cb1_467e_1_f2b96641a5422b292f56d1d0"
carto_url = f"https://basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}.png?key={api_key}"

cx.add_basemap(ax, source=carto_url, zorder=1)

ax.set_axis_off()

legend_elements = [
    Line2D([0], [0], color='#950606', lw=3, label='Dense'),
    Line2D([0], [0], color='#198450', lw=3, label='Low'),
    Line2D([0], [0], color='#ee7600', lw=3, label='Mid')
]

ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=12, title="Area Types", title_fontsize=14)

fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)

plt.savefig(out_dir / "amsterdam_overview.png", dpi=300, bbox_inches="tight")
plt.close()

# for outline_path, tree_path in zip(outline_files, tree_files):
#     area_name = outline_path.stem
#
#     outline = gpd.read_file(outline_path).dissolve().to_crs(epsg=3857)
#     trees = gpd.read_file(tree_path).to_crs(epsg=3857)
#
#     fig, ax = plt.subplots(figsize=(10, 10))
#
#     outline.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2, zorder=2)
#     trees.plot(ax=ax, color="#198450", markersize=10, alpha=0.8, zorder=3)
#
#     minx, miny, maxx, maxy = outline.total_bounds
#     center_x = (minx + maxx) / 2
#     center_y = (miny + maxy) / 2
#     width = maxx - minx
#     height = maxy - miny
#     max_dim = max(width, height)
#
#     buffer = 100
#     half_span = (max_dim / 2) + buffer
#
#     ax.set_xlim(center_x - half_span, center_x + half_span)
#     ax.set_ylim(center_y - half_span, center_y + half_span)
#
#     cx.add_basemap(ax, source=carto_url, zorder=1)
#
#     ax.set_axis_off()
#     ax.set_title(f"Tree Inventory: {area_name}", fontsize=16)
#
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)
#     plt.savefig(out_dir / f"zoomed_{area_name}.png", dpi=300)
#     plt.close()