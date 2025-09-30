import os
import json
import argparse

def compute_average(model_folder: str, base_path: str = "output"):
    """
    Computes the average of each metric across all datasets for a given model.

    Args:
        model_folder (str): Name of the model's folder inside the base_path directory.
        base_path (str): Root directory containing model folders (default: 'output').
    """
    model_path = os.path.join(base_path, model_folder)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model folder '{model_path}' does not exist.")

    metric_sums = {}
    dataset_count = 0

    # Iterate over each dataset folder
    for dataset_name in sorted(os.listdir(model_path)):
        dataset_path = os.path.join(model_path, dataset_name)
        json_file = os.path.join(dataset_path, "test_results.json")
        if os.path.isfile(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Assuming single epoch per file: take the first (and only) entry
            epoch_metrics = next(iter(data.values()))

            # Accumulate metrics
            for metric, value in epoch_metrics.items():
                metric_sums[metric] = metric_sums.get(metric, 0.0) + value

            dataset_count += 1
        else:
            print(f"Warning: 'test_results.json' not found in {dataset_path}")

    if dataset_count == 0:
        print("No valid datasets found for aggregation.")
        return

    # Compute averages
    avg_metrics = {metric: val / dataset_count for metric, val in metric_sums.items()}

    # Display results
    print(f"Average metrics for model '{model_folder}' across {dataset_count} datasets:")
    for metric, avg_value in sorted(avg_metrics.items()):
        print(f"  {metric}: {(avg_value*100):.4f}")


def main():
    compute_average('COATNET_ABN_CF_GAP', 'output')


if __name__ == "__main__":
    main()
