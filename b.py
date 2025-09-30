import os
import json
import math
from statistics import mean, stdev

def calculate_adcc(coherency, complexity, average_drop):
    try:
        return 3 * ((1 / coherency) + (1 / 1 - complexity) + (1 / 1 - average_drop)) ** -1
    except ZeroDivisionError:
        return None  # Skip invalid values

def process_cam_metrics_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    adcc_values = []
    for item in data.values():
        coherency = item.get("coherency")
        complexity = item.get("complexity")
        average_drop = item.get("average_drop")
        adcc = calculate_adcc(coherency, complexity, average_drop)
        if adcc is not None:
            adcc_values.append(adcc)

    if adcc_values:
        adcc_mean = mean(adcc_values)
        adcc_std = stdev(adcc_values) if len(adcc_values) > 1 else 0.0
        return adcc_mean*100, adcc_std*100
    return None, None

def process_all_datasets(root_dir):
    results = {}
    for model_name in os.listdir(root_dir):
        model_path = os.path.join(root_dir, model_name)
        if os.path.isdir(model_path):
            for dataset in os.listdir(model_path):
                dataset_path = os.path.join(model_path, dataset)
                cam_metrics_path = os.path.join(dataset_path, "cam_metrics.json")
                if os.path.exists(cam_metrics_path):
                    adcc_mean, adcc_std = process_cam_metrics_file(cam_metrics_path)
                    if adcc_mean is not None:
                        results[f"{model_name}/{dataset}"] = {
                            "mean_adcc": adcc_mean,
                            "std_adcc": adcc_std
                        }
    return results

if __name__ == "__main__":
    output_folder = "output"  # Change this to your actual path if needed
    results = process_all_datasets(output_folder)
    for dataset, metrics in results.items():
        print(f"{dataset} -> Mean ADCC: {metrics['mean_adcc']:.4f}, Std Dev: {metrics['std_adcc']:.4f}")