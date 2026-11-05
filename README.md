# GEOIT1501-Trees

Automatic tree reconstruction for CFD simulations.

This repository contains a LiDAR-driven workflow for converting urban tree point clouds into 3D mesh models suitable for computational fluid dynamics (CFD) and urban airflow studies created during the Synthesis project of MSc Geomatics at TU Delft. Because a large codebase already existed, the main Python files are divided into three folders;

- the main pipeline in `tree4cfd/` mainly designed by Themis Vargiemezis that filters, segments, and reconstructs individual trees,
- code for automatically downloading AHN and 3DBAG tiles in `CFTree` mainly designed by Noah Alting,
- our newly designed codebase in `synthesis/` which includes a Streamlit application for automated tree reconstruction and a way to determine the density type of an area.

## Project goal

The project turns aerial LiDAR / point-cloud data into realistic tree canopies and trunks, exported as OBJ meshes with local coordinate offsets. These outputs can be used as geometry inputs for CFD simulations and urban wind analysis.

During the synthesis project, different extra segmentation steps were implemented in an attempt to improve the accuracy. On the reconstruction side, alpha wrap was added in order to create simpler meshes resulting in smaller file sizes.

The core pipeline performs the following steps:

1. Load and classify point-cloud tiles.
2. Remove non-vegetation and noisy points.
3. Segment individual trees.
4. Filter non-tree artifacts such as poles or wall-like structures.
5. Remove trees within building footprints.
6. Reconstruct crown and trunk meshes.
7. Export one or more OBJ files plus metadata such as timing and quality scores.

This pipeline was integrated in the user interface, which automated the whole process.

## Repository structure

```text
.
├── README.md
├── environment.yml                # Conda environment definition
├── config.json                    # A general config file
├── synthesis/                     # Newly designed codebase
│   ├── app.py                     # Main UI for Streamlit interface
│   ├── get_data.py                # GeoJSON/AHN preprocessing helpers
│   ├── clip_and_merge.py          # Combining multiple AHN tiles to be exactly the defined AOI
│   ├── determine_lcz_of_area.py   # Use Google Earth Engine to determine urban density type
│   ├── inventory_overlap.py       # Compare detected trees to municipal tree inventory
│   └── ...
├── tree4cfd/                      # Core reconstruction library
│   ├── __main__.py                # Enables `python -m tree4cfd`
│   ├── cli.py                     # Command-line entry points
│   ├── config.py                  # JSON config parsing
│   ├── pipeline.py                # End-to-end orchestration
│   ├── segmentation.py            # Tree segmentation logic
│   ├── crown.py                   # Crown meshing
│   ├── trunk.py                   # Trunk meshing
│   ├── cleaning.py                # Noise filtering and downsampling
│   └── ...
├── CFTree/                        # Supporting file for downloading tiles
├── tests/                         # Project test files
├── tuning-files/                  # Files used for benchmarking the segmentation method
└── resources/                     # Additional support data
```

## Requirements

This project is designed for Python 3.11 and is typically run in a Conda environment. The `environment.yml` file includes the key dependencies, including:

- `streamlit`
- `folium` and `streamlit-folium`
- `geopandas`
- `laspy` with LAS/LAZ support
- `pyvista`, `trimesh`
- `numpy`, `scipy`, `scikit-image`, `shapely`, `pyproj`
- `pandas`, `scikit-learn`

Create the environment and activate it:

```bash
conda env create -f environment.yml
conda activate synthesis-app
```


## Web application usage

Run the interactive web application:

```bash
streamlit run synthesis/app.py
```

From the app, you can:

- draw a polygon on the map, or
- upload a GeoJSON area of interest,
- choose LoD settings,
- optionally include 3D BAG building geometry,
- run the reconstruction pipeline,
- download generated OBJ meshes and preview them in 3D.

## Command-line usage

The core library also exposes a CLI:

```bash
python -m tree4cfd --help
python -m tree4cfd run
```

This is useful when you want to run the pipeline non-interactively or batch process configured areas.

## Configuring the pipeline

The project uses JSON configuration files to define the LiDAR-processing and reconstruction parameters. An example file can be found in the root directory (`config.json`)

Important settings include:

- `paths.tiles_dir` and `paths.output_dir`
- `translation` for local coordinate origin handling
- `lod` level
- trunk and crown meshing options
- output mode and inventory matching

## Data expectations

The standard workflow consumes LiDAR tiles and AOI polygons, typically stored in `DATA/IN/` and written to `DATA/OUT/`.

When running the pipeline using the web application, all files are downloaded and generated automatically. However, when using CLI execution, `.laz` files need to be present inside a `DATA\IN\` folder.

## Output files

After a successful run, the pipeline generates files such as:

- `.obj` mesh files for crown and trunk geometry,
- `scores.json` with accuracy metrics when available,
- `time-summary.json` with execution timing information,
- `offset.txt` recording coordinate offsets,
- optional building meshes.