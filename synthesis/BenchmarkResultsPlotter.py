import sys
from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import griddata


def main():
    folder = Path("DATA/OUT/parameter_tweaking")

    json_files = list(folder.glob("*.json"))

    best_parameters_dense = {}
    best_parameters_low = {}
    best_parameters_mid = {}

    for json_file in json_files:
        if "dense" in json_file.name:
            with open(json_file, "r") as f:
                data = json.load(f)
            perc = 0
            params = {}
            for temp in data:
                if temp.get("F1-Score") > perc:
                    perc = temp.get("F1-Score")
                    params = temp
            best_parameters_dense[json_file.stem] = params

        elif "low" in json_file.name:
            with open(json_file, "r") as f:
                data = json.load(f)
            perc = 0
            params = {}
            for temp in data:
                if temp.get("F1-Score") > perc:
                    perc = temp.get("F1-Score")
                    params = temp
            best_parameters_low[json_file.stem] = params

        elif "mid" in json_file.name:
            with open(json_file, "r") as f:
                data = json.load(f)
            perc = 0
            params = {}
            for temp in data:
                if temp.get("F1-Score") > perc:
                    perc = temp.get("F1-Score")
                    params = temp
            best_parameters_mid[json_file.stem] = params

    print("Best parameters for dense datasets:")
    print(best_parameters_dense)
    print("Best parameters for low datasets:")
    print(best_parameters_low)
    print("Best parameters for mid datasets:")
    print(best_parameters_mid)

        
    for param in ["smooth_sigma", "cell_size", "peak_min_dist_m"]:
        avg = 0
        for temp in best_parameters_dense.values():
            avg+= temp.get(param)
        avg /= len(best_parameters_dense)
        print(f"Average {param} for dense datasets: {avg}")

    for param in ["smooth_sigma", "cell_size", "peak_min_dist_m"]:
        avg = 0
        for temp in best_parameters_low.values():
            avg+= temp.get(param)
        avg /= len(best_parameters_low)
        print(f"Average {param} for low datasets: {avg}")

    for param in ["smooth_sigma", "cell_size", "peak_min_dist_m"]:
        avg = 0
        for temp in best_parameters_mid.values():
            avg+= temp.get(param)
        avg /= len(best_parameters_mid)
        print(f"Average {param} for mid datasets: {avg}")

    avg = 0
    for temp in best_parameters_dense.values():
        avg += temp.get("F1-Score")
    avg /= len(best_parameters_dense)
    print(f"Average F1-Score for dense datasets: {avg}")

    avg = 0
    for temp in best_parameters_low.values():
        avg += temp.get("F1-Score")
    avg /= len(best_parameters_low)
    print(f"Average F1-Score for low datasets: {avg}")

    avg = 0
    for temp in best_parameters_mid.values():
        avg += temp.get("F1-Score")
    avg /= len(best_parameters_mid)
    print(f"Average F1-Score for mid datasets: {avg}")


    for file in json_files:
        plotted_data = {}
        with open(file, "r") as f:
            data = json.load(f)
        for temp in data:
            if (temp["peak_min_dist_m"], temp["cell_size"]) in plotted_data:
                plotted_data[(temp["peak_min_dist_m"], temp["cell_size"])] += temp["F1-Score"]
                plotted_data[(temp["peak_min_dist_m"], temp["cell_size"])] /= 2
            else:
                plotted_data[(temp["peak_min_dist_m"], temp["cell_size"])] = temp["F1-Score"]

        x = np.array([])
        y = np.array([])
        z = np.array([])
        for key in plotted_data:
            x = np.append(x, key[0])
            y = np.append(y, key[1])
            z = np.append(z, plotted_data[key])

    # Create heat map
        xi = np.linspace(x.min(), x.max(), 200)
        yi = np.linspace(y.min(), y.max(), 200)
        Xi, Yi = np.meshgrid(xi, yi)

        # Interpolate your measurements onto the grid
        Zi = griddata(
            (x, y),
            z,
            (Xi, Yi),
            method="cubic"
        )

        # Plot
        plt.figure(figsize=(8, 6))

        plt.contourf(Xi, Yi, Zi, levels=50, cmap="viridis")

        # Show original measurement locations
        plt.scatter(x, y, c=z, edgecolor="black", cmap="viridis")

        plt.xlabel("Peak Minimum Distance (m)")
        plt.ylabel("Cell Size")
        plt.colorbar(label="F1-Score")
        plt.title(f"Interpolated Heat Map for Accuracy for {file.stem.replace('benchmark_results_', '')}")

        plt.show()
        

if __name__ == "__main__":
    sys.exit(main())

