import json
import glob
import os
import matplotlib.pyplot as plt

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

directory_path = str(root_dir) + '/DATA/OUT/Sensitivity/delft'
file_pattern = 'sensitivity_results_*.json'
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'D', 'v', 'P']

def plot_sensitivity_results():
    # Find all JSON files
    file_paths = glob.glob(os.path.join(directory_path, file_pattern))
    if not file_paths:
        print("No files found matching the pattern.")
        return

    all_data = {}
    parameters = []

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        area_name = area_name = filename.replace('sensitivity_results_', '').replace('.json', '')

        with open(filepath, 'r') as f:
            data = json.load(f)
            all_data[area_name] = data

            if not parameters:
                parameters = list(data.keys())

    num_params = len(parameters)
    cols = 5
    rows = (num_params + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 8), sharey=True)
    axes = axes.flatten()

    # Plot each parameter in its own subplot
    for idx, param in enumerate(parameters):
        ax = axes[idx]

        for area_idx, (area_name, area_data) in enumerate(all_data.items()):
            if param in area_data:
                # Extract values and scores for this specific parameter and area
                values = [entry['value'] for entry in area_data[param]]
                f1_scores = [entry['F1-Score'] for entry in area_data[param]]

                # Plot the line
                ax.plot(values, f1_scores, label=area_name,
                        color=colors[area_idx % len(colors)],
                        marker=markers[area_idx % len(markers)],
                        linewidth=2, markersize=6)

        ax.set_title(param, fontsize=12, fontweight='bold')
        ax.set_xlabel('Value')
        if idx % cols == 0:
            ax.set_ylabel('F1-Score')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Hide any unused subplots if parameters < rows * cols
    for i in range(num_params, len(axes)):
        fig.delaxes(axes[i])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=12, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    plt.savefig(os.path.join(directory_path, 'sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_sensitivity_results()