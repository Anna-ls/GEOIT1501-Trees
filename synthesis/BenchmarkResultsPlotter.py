import sys
from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import math
import re

from matplotlib.colors import Normalize


import json
from pathlib import Path


def main():
    folder = Path("combined")
    json_files = list(folder.glob("*.json"))

    best_parameters = {
        "dense": {},
        "low": {},
        "mid": {},
    }

    # Find the best-performing parameter combination per dataset
    for json_file in json_files:
        filename = json_file.name.lower()

        urban_class = next(
            (cls for cls in ["dense", "low", "mid"]
             if cls in filename),
            None,
        )

        if urban_class is None:
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            continue

        best = max(
            data,
            key=lambda temp: temp.get("F1-Score", float("-inf")),
        )

        best_parameters[urban_class][json_file.stem] = best

    parameters = [
        "smooth_sigma",
        "cell_size",
        "peak_min_dist_m",
    ]

    # Calculate weighted averages per urban class
    for urban_class, datasets in best_parameters.items():
        print(f"\nBest parameters for {urban_class} datasets:")
        print(datasets)

        if not datasets:
            print("No datasets found.")
            continue

        total_weight = sum(
            result["F1-Score"]
            for result in datasets.values()
        )

        if total_weight <= 0:
            print("Cannot calculate weighted averages: "
                  "total F1-score weight is zero or negative.")
            continue

        for param in parameters:
            weighted_avg = sum(
                result[param] * result["F1-Score"]
                for result in datasets.values()
            ) / total_weight

            print(
                f"Weighted average {param} for "
                f"{urban_class} datasets: {weighted_avg:.4f}"
            )

        # Average F1-score across the best results
        avg_f1 = sum(
            result["F1-Score"]
            for result in datasets.values()
        ) / len(datasets)

        print(
            f"Average best F1-Score for "
            f"{urban_class} datasets: {avg_f1:.4f}"
        )


    datasets = plot_benchmark_heatmaps(json_files)

def plot_benchmark_heatmaps(
    json_files,
    x_col = "cell_size",
    y_col = "smooth_sigma",
    other_col = "peak_min_dist_m",
    output_dir="benchmark_plots",
    class_order=("dense", "mid", "low"),
    vmin=None,
    vmax=None,
    grid_size=200,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_order = list(class_order)

    # Consistent styling and colours across all figures
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    cmap = "viridis"

    # --------------------------------------------------
    # 1. Load data and calculate proper averages
    # --------------------------------------------------
    datasets = {}

    for file in json_files:
        file = Path(file)

        with open(file, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        required = {x_col, y_col, "F1-Score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{file.name} is missing columns: {missing}"
            )

        # Average all repeated measurements for each parameter pair
        df = (
            df.groupby([x_col, y_col, other_col], as_index=False)
            ["F1-Score"].mean()
        )

        df = (df.loc[df.groupby([x_col, y_col])["F1-Score"].idxmax()].reset_index(drop=True))

        df = df[[x_col, y_col, "F1-Score"]]

        df = df.rename(columns={
            x_col: "x",
            y_col: "y",
            "F1-Score": "score",
        })

        

        # Detect class from the filename
        match = re.search(
            r"(?:^|[_-])(dense|mid|low)(?:[_-]|$)",
            file.stem.lower(),
        )
        urban_class = match.group(1) if match else "unknown"

        datasets[file.stem] = {
            "file": file,
            "class": urban_class,
            "data": df,
        }

    if not datasets:
        raise ValueError("No JSON files were supplied.")

    # --------------------------------------------------
    # 2. Establish shared x/y ranges and interpolation grid
    # --------------------------------------------------
    all_data = pd.concat(
        [item["data"] for item in datasets.values()],
        ignore_index=True,
    )

    # Automatically determine the colour scale from all benchmark scores
    all_scores = all_data["score"]

    if vmin is None:
        vmin = all_scores.min()

    if vmax is None:
        vmax = all_scores.max()

    norm = Normalize(vmin=vmin, vmax=vmax)

    xmin, xmax = all_data["x"].min(), all_data["x"].max()
    ymin, ymax = all_data["y"].min(), all_data["y"].max()

    def padded_range(low, high):
        if low == high:
            pad = max(abs(low) * 0.05, 0.5)
            return low - pad, high + pad
        pad = (high - low) * 0.03
        return low - pad, high + pad

    xlim = padded_range(xmin, xmax)
    ylim = padded_range(ymin, ymax)

    xi = np.linspace(xmin, xmax, grid_size)
    yi = np.linspace(ymin, ymax, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    def interpolate(df):
        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        z = df["score"].to_numpy(dtype=float)

        points = np.column_stack([x, y])
        grid_points = (Xi, Yi)

        # Cubic interpolation needs sufficient non-collinear points.
        # Fall back gracefully if the data is too sparse.
        if len(df) >= 4:
            try:
                Zi = griddata(
                    points, z, grid_points, method="cubic"
                )
            except Exception:
                Zi = griddata(
                    points, z, grid_points, method="linear"
                )
        else:
            Zi = griddata(
                points, z, grid_points, method="linear"
            )

        # Fill outside the convex hull with nearest measurements
        # to avoid blank areas in the heatmap.
        if np.isnan(Zi).any():
            nearest = griddata(
                points, z, grid_points, method="nearest"
            )
            Zi = np.where(np.isnan(Zi), nearest, Zi)

        return x, y, z, Zi

    # --------------------------------------------------
    # 3. Shared plotting function
    # --------------------------------------------------
    def draw_heatmap(ax, item, show_ylabel=True):
        df = item["data"]
        x, y, z, Zi = interpolate(df)

        # Smooth background with a fixed colour scale
        filled = ax.contourf(
            Xi, Yi, Zi,
            levels=np.linspace(vmin, vmax, 51),
            cmap=cmap,
            norm=norm,
            extend="neither",
        )

        # Optional contour lines to make score differences clearer
        ax.contour(
            Xi, Yi, Zi,
            levels=np.linspace(vmin, vmax, 6),
            colors="white",
            linewidths=0.45,
            alpha=0.45,
        )

        # Actual benchmark locations, outlined for visibility
        ax.scatter(
            x, y,
            c=z,
            cmap=cmap,
            norm=norm,
            s=48,
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_col.replace("_", " ").title() if show_ylabel else "")
        ax.grid(False)
        ax.set_axisbelow(True)

        return filled

    # --------------------------------------------------
    # 4. Individual plots
    # --------------------------------------------------
    for name, item in datasets.items():
        fig, ax = plt.subplots(figsize=(8, 6))

        draw_heatmap(ax, item)

        fig.suptitle(
            f"F1-Score — {name.replace('benchmark_results_', '')}",
            x=0.12,
            ha="left",
            fontweight="semibold",
        )

        fig.subplots_adjust(
            left=0.12, right=0.86, bottom=0.12, top=0.88
        )

        # Identical scale and colour bar on every figure
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, pad=0.035)
        cbar.set_label("F1-Score", rotation=90, labelpad=12)
        cbar.set_ticks(np.linspace(vmin, vmax, 6))

        fig.savefig(
            output_dir / f"{name}_heatmap.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)

    # --------------------------------------------------
    # 5. Combined figure: dense | mid | low
    # --------------------------------------------------
    available_classes = [
        cls for cls in class_order
        if any(item["class"] == cls for item in datasets.values())
    ]

    if available_classes:
        fig, axes = plt.subplots(
            1,
            len(available_classes),
            figsize=(6 * len(available_classes), 5.5),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes = axes.flatten()

        for ax, cls in zip(axes, available_classes):
            class_items = [
                item for item in datasets.values()
                if item["class"] == cls
            ]

            # If multiple files belong to the same class, combine
            # their scores by parameter pair using a mean.
            combined = pd.concat(
                [item["data"] for item in class_items],
                ignore_index=True,
            )
            combined = (
                combined.groupby(["x", "y"], as_index=False)["score"]
                .mean()
            )

            combined_item = {
                "data": combined,
                "class": cls,
            }

            draw_heatmap(ax, combined_item)

            ax.set_title(
                cls.capitalize(),
                fontweight="semibold",
                pad=12,
            )

        fig.suptitle(
            "F1-Score Comparison Across Urban Classes",
            fontweight="semibold",
            y=1.02,
        )

        fig.supxlabel(x_col.replace("_", " ").title(), y=0.02)
        fig.supylabel(y_col.replace("_", " ").title(), x=0.01)

        fig.subplots_adjust(
            left=0.08,
            right=0.88,
            bottom=0.13,
            top=0.84,
            wspace=0.12,
        )

        # One shared colour bar for the entire comparison
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=axes.tolist(),
            location="right",
            fraction=0.025,
            pad=0.025,
        )
        cbar.set_label("F1-Score", labelpad=10)
        cbar.set_ticks(np.linspace(vmin, vmax, 6))

        combined_path = output_dir / "combined_class_comparison.png"
        fig.savefig(
            combined_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.show()
        plt.close(fig)

    # --------------------------------------------------
    # 6. Combined figure: every JSON file
    # --------------------------------------------------
    n_plots = len(datasets)
    ncols = min(3, n_plots)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, (name, item) in zip(axes, datasets.items()):
        draw_heatmap(ax, item)

        # Use a readable title derived from the filename
        panel_title = name.replace("benchmark_results_", "")
        panel_title = panel_title.replace("_", " ").title()

        ax.set_title(
            panel_title,
            fontweight="semibold",
            pad=10,
        )

    # Remove unused panels
    for ax in axes[n_plots:]:
        fig.delaxes(ax)

    fig.suptitle(
        "F1-Score Comparison Across All Benchmark Results",
        fontweight="semibold",
        y=1.01,
    )

    fig.supxlabel(x_col.replace("_", " ").title(), y=0.02)
    fig.supylabel(y_col.replace("_", " ").title(), x=0.01)

    fig.subplots_adjust(
        left=0.09,
        right=0.88,
        bottom=0.10,
        top=0.91,
        wspace=0.15,
        hspace=0.28,
    )

    # One shared colour bar for all files
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes[:n_plots].tolist(),
        location="right",
        fraction=0.025,
        pad=0.025,
    )
    cbar.set_label("F1-Score", labelpad=10)
    cbar.set_ticks(np.linspace(vmin, vmax, 6))

    all_path = output_dir / "all_benchmarks_comparison.png"

    fig.savefig(
        all_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()
    plt.close(fig)

    print(f"All-files comparison saved to: {all_path.resolve()}")


    print(f"Plots saved to: {output_dir.resolve()}")

    return datasets

        

if __name__ == "__main__":
    sys.exit(main())

