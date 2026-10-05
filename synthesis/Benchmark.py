from functools import wraps
import time
import subprocess
import psutil
import pandas as pd
import argparse
from tree4cfd.inspect_laz import profile_classes
from tree4cfd.io import find_tiles
from tree4cfd.validation import validation_pipeline
import json
import matplotlib.pyplot as plt
import numpy as np
from itertools import product




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
            "Difference": [],
            "Percentage": []
        }

        self.parameters_results = {
            "tweaked_parameter": [],
            "results": []
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

            
            peak_memory_mb = peak_memory / (1024 ** 2)
            avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
            total_cores = psutil.cpu_count(logical=True)
            normalized_cpu = avg_cpu / total_cores if total_cores else avg_cpu
            success = (process.returncode == 0)
            status_str = "SUCCESS" if success else f"FAILED (Code {process.returncode})"

            

            with open(f"config_files_segmentation/{config_file}", "r") as f:
                json_file = json.load(f)

            tiles = find_tiles(json_file["paths"]["tiles_dir"])

            # Validation values, reference and detected trees counts
            with open(f"./parameters_segmentation.json", "r") as f:
                params = json.load(f)

            ref, det = validation_pipeline(file, params)
            n_ref = len(ref) if ref is not None else 0
            n_det = len(det) if det is not None else 0
            diff = n_det - n_ref
            pct = (diff / n_ref * 100) if n_ref else float("nan")
            elapsed_time = time.perf_counter() - start_time

            print(
                            f"{config_file} [{status_str}]: "
                            f"{elapsed_time:.2f}s, "
                            f"{peak_memory_mb:.2f} MB, "
                            f"CPU: {normalized_cpu:.1f}%"
                        )

            stats = profile_classes(tiles)
            total_points = sum(s["count"] for s in stats.values())
            self.results["Name"].append(config_file)
            self.results["Status"].append(status_str)
            self.results["Process Time (s)"].append(round(elapsed_time, 2))
            self.results["Memory Usage (MB)"].append(round(peak_memory_mb, 2))
            self.results["CPU Usage (%)"].append(round(normalized_cpu, 1))
            self.results["Total Points"].append(total_points)
            self.results["Reference Trees"].append(n_ref)
            self.results["Detected Trees"].append(n_det)
            self.results["Difference"].append(abs(diff))
            self.results["Percentage"].append(pct)
            # self.results["tweaked_parameter"].append(params.get(param, None))

            self.parameters_results["results"][-1].append(abs(pct))

            


            if self.show_progress or not success:
                print("stdout:", stdout)
                print("stderr:", stderr)

            return process

        return inner

    def run(self):
        self.tune_parameters()  # Initialize with default parameters
        with open(self.parameter_tweak_file, "r") as f:
            tweak_params = json.load(f)

        self.parameters_results["tweaked_parameter"] = tweak_params["Tweaked Parameters"]
        self.parameters_results["results"] = []
        
        for file in self.files:
            i=0
            self.parameters_results["results"] = []
            for combination in product(*[np.arange(tweak_params["Ranges"][param][0], tweak_params["Ranges"][param][1], tweak_params["Ranges"][param][2]) for param in tweak_params["Tweaked Parameters"]]):
                param_dict = dict(zip(tweak_params["Tweaked Parameters"], combination))
                print(f"Inspecting files with parameters: {param_dict}...")
                self.tune_parameters(**param_dict)
                self.parameters_results["results"].append(list(combination))
            
                self.inspect(file)
                i+=1

            self.plot_results(file)


        # for param_index in range(len(tweak_params["Tweaked Parameters"])):
        #     param = tweak_params["Tweaked Parameters"][param_index]
        #     for i in np.arange(tweak_params["Ranges"][param][0], tweak_params["Ranges"][param][1], tweak_params["Ranges"][param][2]):
        #         print(f"Inspecting files with {param}={i}...")
        #         for file in self.files:
        #             self.tune_parameters(**{param: i})
        #             self.inspect(file, param=param)
        #     self.plot_results(param)
        #     self.results = {key: [] for key in self.results}  # Reset results for the next parameter
        #     self.tune_parameters()  # Reset to default parameters after each parameter sweep

        # print("\nReconstructing files...")
        # for file in self.files:
        #     self.reconstruct(file)

        
    
    @statistics
    def inspect(self, file):
        print(f"Running inspection with {file}...")
        return subprocess.Popen(
            ["python", "-m", "tree4cfd", "inspect", "--config", f"config_files_segmentation/{file}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    @statistics
    def reconstruct(self, config_file):
        print(f"Running reconstruction with {config_file}...")
        return subprocess.Popen(
            ["python", "-m", "tree4cfd", "run", "--config", f"config_files_segmentation/{config_file}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def tune_parameters(self, cell_size=1.0, smooth_sigma=2.6, min_height=2.5, peak_min_dist_m=3.5, min_tree_points=30, max_elongation=3, max_offset_ratio=1,
                        resolve_multi_trees="True", num_angles=2, bin_size=0.5, min_peak_dist=3.5, d_euclidean_thresh=2.2, d_margin_thresh=1.5):
        parameters = {    
                        "cell_size": cell_size,           
                        "smooth_sigma": smooth_sigma,       
                        "min_height": min_height,          
                        "peak_min_dist_m": peak_min_dist_m,     
                        "min_tree_points": min_tree_points,      
                        "max_elongation": max_elongation,        
                        "max_offset_ratio": max_offset_ratio,    
                        "resolve_multi_trees": resolve_multi_trees,

                        "num_angles": num_angles,
                        "bin_size": bin_size,
                        "min_peak_dist": min_peak_dist,
                        "d_euclidean_thresh": d_euclidean_thresh,
                        "d_margin_thresh": d_margin_thresh
                    }
        with open("parameters_segmentation.json", "w") as f:
            json.dump(parameters, f, indent=4)

    def plot_results(self, file):
        print(self.parameters_results)
        df = pd.DataFrame(self.parameters_results["results"], columns=self.parameters_results["tweaked_parameter"] + ["Difference"])
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

    args = parser.parse_args()
    with open(args.file, "r") as f:
        config_files = [line.strip() for line in f if line.strip()]

    benchmark = Benchmark(config_files, show_progress=args.show_progress, parameter_tweak_file=args.parameter_tweak_file)
    benchmark.run()
    benchmark.print_results()
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())