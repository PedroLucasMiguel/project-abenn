import os
import json
import numpy as np

def compile_results(json_path_1, json_path_2) -> None:

    f = open(json_path_1, "r")
    js1 = json.load(f)
    f.close()

    f = open(json_path_2, "r")
    js2 = json.load(f)
    f.close()

    n_samples = len(js1.keys())
    n_metrics = len(js1[list(js1.keys())[0]])

    results = np.zeros(shape=(n_metrics, 2))

    for k in js1.keys():
        j = 0
        for m in js1[k].keys():

            if m == "coherency" or m == "adcc":
                if js1[k][m] > js2[k][m]:
                    results[j][0] += 1
                elif js1[k][m] == js2[k][m]:
                    results[j][0] += 1
                    results[j][1] += 1
                else:
                    results[j][1] += 1
                j += 1

            else:
                if js1[k][m] < js2[k][m]:
                    results[j][0] += 1
                elif js1[k][m] == js2[k][m]:
                    results[j][0] += 1
                    results[j][1] += 1
                else:
                    results[j][1] += 1
                j += 1

    m_labels = {0: "Coherency", 1: "Complexity", 2: "Average Drop", 3: "ADCC"}

    i = 0
    for r in results:
        if r[0] > r[1]:
            print(f"{m_labels[i]} in the FIRST model is {((r[0]/n_samples) - (r[1]/n_samples))*100}% better")
        else:
            print(f"{m_labels[i]} in the SECOND model is {((r[1]/n_samples) - (r[0]/n_samples))*100}% better")
        i += 1

    pass


if __name__ == "__main__":
    models_to_compare = ['ABN', 'RESNET_ABN']

    folders = os.listdir(f"../output/{models_to_compare[0]}")

    #compile_results(f"../output/{models_to_compare[0]}/{'CR_2_CV'}/cam_metrics.json", 
    #                f"../output/{models_to_compare[1]}/{'CR_2_CV'}/cam_metrics.json")
    
    for f in folders:
        print(f"{'='*10}")
        print(f"Results from folder: {f}")

        compile_results(f"../output/{models_to_compare[0]}/{f}/cam_metrics.json", 
                        f"../output/{models_to_compare[1]}/{f}/cam_metrics.json")
        print(f"{'='*10}")
    