import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from functools import wraps
import time
import subprocess
import psutil
import pandas as pd
import argparse
import json
import numpy as np
from itertools import product

from synthesis.validation import validation_pipeline

class Benchmark:
    def __init__(self, files, show_progress=False, parameter_tweak_file=None):
        self.files = files
        self.show_progress = show_progress
        self.parameter_tweak_file = parameter_tweak_file
        self.results = {
            "Name": [], 
            "Status": [], 
            "Process Time (s)": [], 
            "Memory Usage (MB)": [],
            "CPU Usage (%)": [],
            "Total Points": [],
            "Reference Trees": [],
            "Detected Trees": [],
            "Precision": [],
            "Recall": [],
            "F1-Score": []
        }

        self.parameters_results = {
            "tweaked_parameter": [],
            "results": []
        }

        self.base_params = {
            "cell_size": 1.0,
            "smooth_sigma": 1.55,
            "min_height": 2.5,
            "peak_min_dist_m": 3.0,
            "min_tree_points": 60,
            "max_elongation": 3.0,
            "max_offset_ratio": 1.5,
            "resolve_multi_trees": True,
            "num_angles": 4,
            "bin_size": 0.5,
            "min_peak_dist_3d": 3.0,
            "d_euclidean_thresh": 2.5,
            "d_margin_thresh": 1.5
        }

    @staticmethod
    def statistics(func):
        @wraps(func)
        def inner(self, file, *args, **kwargs):
            start_time = time.perf_counter()

            config_file = f"config_{file}.json"
            process = func(self, config_file, *args, **kwargs)

            ps_process = psutil.Process(process.pid)
            peak_memory = 0
            cpu_samples = []

            # Monitor the subprocess
            while process.poll() is None:
                try:
                    memory = ps_process.memory_info().rss
                    peak_memory = max(peak_memory, memory)
                    
                    # Capture CPU usage (non-blocking interval)
                    cpu = ps_process.cpu_percent(interval=None)
                    if cpu > 0:
                        cpu_samples.append(cpu)
                except psutil.NoSuchProcess:
                    break

                time.sleep(0.05)

            # Wait until finished and grab stdout/stderr
            stdout, stderr = process.communicate()

            # Computational metrics
            peak_memory_mb = peak_memory / (1024 ** 2)
            avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
            total_cores = psutil.cpu_count(logical=True)
            normalized_cpu = avg_cpu / total_cores if total_cores else avg_cpu
            success = (process.returncode == 0)
            status_str = "SUCCESS" if success else f"FAILED (Code {process.returncode})"

            with open(root_dir / "config_files_segmentation" / config_file, "r") as f:
                json_file = json.load(f)

            # Validation values, reference and detected trees counts
            with open(f"./parameters_segmentation.json", "r") as f:
                params = json.load(f)

            # Unpack spatial validation metrics
            ref, det, precision, recall, f1, total_points = validation_pipeline(file, params)
            n_ref = len(ref) if ref is not None else 0
            n_det = len(det) if det is not None else 0
            elapsed_time = time.perf_counter() - start_time

            print(
                            f"{config_file} [{status_str}]: "
                            f"{elapsed_time:.2f}s, "
                            f"{peak_memory_mb:.2f} MB, "
                            f"CPU: {normalized_cpu:.1f}%"
                        )

            self.results["Name"].append(config_file)
            self.results["Status"].append(status_str)
            self.results["Process Time (s)"].append(round(elapsed_time, 2))
            self.results["Memory Usage (MB)"].append(round(peak_memory_mb, 2))
            self.results["CPU Usage (%)"].append(round(normalized_cpu, 1))
            self.results["Total Points"].append(total_points)
            self.results["Reference Trees"].append(n_ref)
            self.results["Detected Trees"].append(n_det)

            self.results["Precision"].append(round(precision, 3))
            self.results["Recall"].append(round(recall, 3))
            self.results["F1-Score"].append(round(f1, 3))

            if len(self.parameters_results["results"]) > 0:
                self.parameters_results["results"][-1].append(round(f1, 3))

            if self.show_progress or not success:
                print("stdout:", stdout)
                print("stderr:", stderr)

            return process
        return inner

    def run_sensitivity(self):
        print("\n Starting Sensitivity Analysis...")
        with open(self.parameter_tweak_file, "r") as f:
            tweak_params = json.load(f)

        for file in self.files:
            sensitivity_export = {}

            for param in tweak_params["Tweaked Parameters"]:
                print(f"\n Isolating Parameter: {param}...")
                sensitivity_export[param] = []

                # Create the sweep range for just this parameter
                start, stop, step = tweak_params["Ranges"][param]
                test_values = np.arange(start, stop, step)

                for val in test_values:
                    #Keep everything at baseline, only overwrite the active parameter
                    current_params = self.base_params.copy()
                    current_params[param] = val

                    self.tune_parameters(**current_params)
                    self.inspect(file)

                    # Extract F1-score from the latest run in self.results
                    latest_f1 = self.results["F1-Score"][-1]
                    sensitivity_export[param].append({"value": round(val, 3), "F1-Score": latest_f1})

            export_path = root_dir / "DATA" / "OUT" / "Sensitivity" / f"sensitivity_results_{file}.json"

            with open(export_path, "w") as f:
                json.dump(sensitivity_export, f, indent=4)
            print(f"\nSensitivity analysis for {file} saved to {export_path}")

    def run(self):
        self.tune_parameters(**self.base_params)  # Initialize with default parameters
        with open(self.parameter_tweak_file, "r") as f:
            tweak_params = json.load(f)

        self.parameters_results["tweaked_parameter"] = tweak_params["Tweaked Parameters"]
        self.parameters_results["results"] = []
        
        for file in self.files:
            self.parameters_results["results"] = []
            param_ranges = [np.arange(tweak_params["Ranges"][param][0], tweak_params["Ranges"][param][1],
                                      tweak_params["Ranges"][param][2]) for param in tweak_params["Tweaked Parameters"]]

            for combination in product(*param_ranges):
                param_dict = dict(zip(tweak_params["Tweaked Parameters"], combination))
                print(f"Inspecting files with parameters: {param_dict}...")

                current_params = self.base_params.copy()
                current_params.update(param_dict)

                self.tune_parameters(**current_params)
                self.parameters_results["results"].append(list(combination))
            
                self.inspect(file)

            self.plot_results(file)

    @statistics
    def inspect(self, file):
        print(f"Running inspection with {file}...")
        return subprocess.Popen(
            ["python", "-m", "tree4cfd", "inspect", "--config", str(root_dir / "config_files_segmentation" / file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def tune_parameters(self, **kwargs):
        with open("parameters_segmentation.json", "w") as f:
            json.dump(kwargs, f, indent=4)

    def plot_results(self, file):
        print(self.parameters_results)
        df = pd.DataFrame(self.parameters_results["results"], columns=self.parameters_results["tweaked_parameter"] + ["F1-Score"])
        df.to_json(f"benchmark_results_{file}.json", orient="records", indent=4)
        print(df)

    def print_results(self):
        print("\n--- Benchmark Summary ---")
        df = pd.DataFrame(self.results)
        print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the configuration file list")
    parser.add_argument("parameter_tweak_file", help="Path to the parameter tweak file")
    parser.add_argument("-p", "--show-progress", action="store_true", help="Show progress of each benchmark run")
    parser.add_argument("-s", "--sensitivity", action="store_true", help="Run a One-At-a-Time sensitivity analysis instead of a full grid search")

    args = parser.parse_args()
    with open(args.file, "r") as f:
        config_files = [line.strip() for line in f if line.strip()]

    benchmark = Benchmark(config_files, show_progress=args.show_progress, parameter_tweak_file=args.parameter_tweak_file)

    if args.sensitivity:
        benchmark.run_sensitivity()

    else:
        benchmark.run()

    benchmark.print_results()
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())