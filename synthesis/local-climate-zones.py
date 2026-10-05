import ee
import geemap

# Initialize Earth Engine
ee.Initialize()

# 1. Define Netherlands boundary
netherlands = ee.FeatureCollection("FAO/GAUL/2015/level0") \
    .filter(ee.Filter.eq("ADM0_NAME", "Netherlands"))

# 2. Load LCZ collection, mosaic, and clip to boundary
lcz = ee.ImageCollection("RUB/RUBCLIM/LCZ/global_lcz_map/latest") \
    .mosaic() \
    .clip(netherlands)

# 3. Visualization parameters (Classes 1–17)
palette = [
    "8c0000",  # LCZ 1: Compact high-rise
    "d10000",  # LCZ 2: Compact mid-rise
    "ff0000",  # LCZ 3: Compact low-rise
    "bf4d00",  # LCZ 4: Open high-rise
    "ff6600",  # LCZ 5: Open mid-rise
    "ff9955",  # LCZ 6: Open low-rise
    "faee05",  # LCZ 7: Lightweight low-rise
    "bcbcbc",  # LCZ 8: Large low-rise
    "ffccaa",  # LCZ 9: Sparsely built
    "555555",  # LCZ 10: Heavy industry
    "006a00",  # LCZ A: Dense trees
    "00aa00",  # LCZ B: Scattered trees
    "648525",  # LCZ C: Bush, scrub
    "b9db79",  # LCZ D: Low plants
    "000000",  # LCZ E: Bare rock or paved
    "fbf7ae",  # LCZ F: Bare soil or sand
    "6a6aff",  # LCZ G: Water
]

vis_params = {
    "bands": ["LCZ_Filter"],
    "min": 1,
    "max": 17,
    "palette": palette,
}

# 4. Interactive Map
Map = geemap.Map()
Map.centerObject(netherlands, 8)
Map.addLayer(lcz, vis_params, "Netherlands LCZ (Filtered)")
Map.addLayer(
    netherlands.style(fillColor="00000000", color="222222", width=1.5),
    {},
    "Netherlands Border",
)
Map