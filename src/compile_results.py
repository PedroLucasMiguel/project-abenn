import os
import json
import numpy as np

toMean = np.zeros(shape=(5, 4))

def compile_result_to_table(json_path, dt_name, file, i) -> None:
    f = open(json_path, "r")
    js = json.load(f)
    f.close()

    n_samples = len(js.keys())
    n_metrics = len(js[list(js.keys())[0]])
    sumns = np.zeros(shape=(n_metrics, 1), dtype=np.float64)

    for k in js.keys():
        j = 0
        for m in js[k].keys():
            sumns[j] += js[k][m]
            j += 1

    m_labels = {0: "Coherency", 1: "Complexity", 2: "Average Drop", 3: "ADCC"}
    toMean[i][0] = round(((sumns[0][0] / n_samples)*100), 2)
    toMean[i][1] = round(((sumns[1][0] / n_samples)*100), 2)
    toMean[i][2] = round(((sumns[2][0] / n_samples)*100), 2)
    toMean[i][3] = round(((sumns[3][0] / n_samples)*100), 2)

    file.write(f'{dt_name} & {round(((sumns[0][0] / n_samples)*100), 2)} & {round(((sumns[1][0] / n_samples)*100), 2)} & {round(((sumns[2][0] / n_samples)*100), 2)} & {round(((sumns[3][0] / n_samples)*100), 2)} \\\\\n')

    # for r in sumns:
        
    #     # print(
    #     #     f"{m_labels[i]} in the {'FIRST' if r[0] > r[1] else 'SECOND'} model is better in {abs((((r[0] / n_samples) - (r[1] / n_samples)) * 100)):2f}% more cases")
    #     print(f"{m_labels[i]} is {(sumns[i][0] / n_samples)*100}")

    #     i += 1


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
    sumns = np.zeros(shape=(n_metrics, 2), dtype=np.float64)

    for k in js1.keys():
        j = 0
        for m in js1[k].keys():

            sumns[j][0] += js1[k][m]
            sumns[j][1] += js2[k][m]

            if m == "coherency" or m == "adcc":
                if js1[k][m] > js2[k][m]:
                    results[j][0] += 1
                elif js1[k][m] == js2[k][m]:
                    results[j][0] += 1
                    results[j][1] += 1
                else:
                    results[j][1] += 1

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
        # print(
        #     f"{m_labels[i]} in the {'FIRST' if r[0] > r[1] else 'SECOND'} model is better in {abs((((r[0] / n_samples) - (r[1] / n_samples)) * 100)):2f}% more cases")
        print(
            f"{m_labels[i]} is {(sumns[i][0] / n_samples)*100} | {(sumns[i][1] / n_samples)*100}\n")

        i += 1

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
    sumns = np.zeros(shape=(n_metrics, 2), dtype=np.float64)

    for k in js1.keys():
        j = 0
        for m in js1[k].keys():

            sumns[j][0] += js1[k][m]
            sumns[j][1] += js2[k][m]

            if m == "coherency" or m == "adcc":
                if js1[k][m] > js2[k][m]:
                    results[j][0] += 1
                elif js1[k][m] == js2[k][m]:
                    results[j][0] += 1
                    results[j][1] += 1
                else:
                    results[j][1] += 1

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
        # print(
        #     f"{m_labels[i]} in the {'FIRST' if r[0] > r[1] else 'SECOND'} model is better in {abs((((r[0] / n_samples) - (r[1] / n_samples)) * 100)):2f}% more cases")
        print(
            f"{m_labels[i]} is {(sumns[i][0] / n_samples)*100} | {(sumns[i][1] / n_samples)*100}\n")

        i += 1


if __name__ == "__main__":
    #models_to_compare = ['RESNET50_ABN_CF_GAP', 'RESNET50']
    #models_to_compare = ['DENSENET201_ABN_CF_GAP', 'DENSENET201']
    #models_to_compare = ['CONVNEXT_ABN_CF_GAP', 'CONVNEXT_SMALL']
    #models_to_compare = ['EFFICIENTNET_ABN_CF_GAP', 'EFFICIENTNET',]
    #models_to_compare = ['COATNET_ABN_CF_GAP', 'COATNETB0']
    #models_to_compare = ['RESNEXT50_ABN_CF_GAP', 'RESNEXT50']
    models_to_compare = ['UNIFORMER_ABN_CF_GAP', 'UNIFORMER_BASELINE']

    MODEL = 0
    folders = os.listdir(f"../output/{models_to_compare[MODEL]}")

    headers = [
        '\\begin{table}[H]\n',
        '\\caption{Percentage values of coherence (CO), complexity (COM), drop in confidence (CD) and ADCC for the X model, considering all datasets.}\n',
        '\\begin{tabularx}{\\textwidth}{CCCCC}\n',
        '\\toprule',
        '\\textbf{Dataset} & CO$\\uparrow$ & COM$\\downarrow$ & AD$\\downarrow$ & ADCC$\\uparrow$ \\\\',
        '\\midrule\n'
    ]
    footer = [
        #'\\midrule\n',
        #'\\textbf{Mean} & X & X & X & X \\\\\n',
        '\\bottomrule\n',
        '\\end{tabularx}\n',
        '\\end{table}'
    ]

    i = 0
    with open(f'../tables/{models_to_compare[MODEL]}.txt', 'w') as file:
        file.writelines(headers)
        for f in folders:
            compile_result_to_table(f"../output/{models_to_compare[MODEL]}/{f}/cam_metrics.json", f.split('_')[0], file, i)
            i += 1
        file.write('\\midrule\n')
        means = np.mean(toMean, axis=0)
        file.write(f'\\textbf{{Mean}} & {round(means[0], 2)} & {round(means[1], 2)} & {round(means[2], 2)} & {round(means[3], 2)} \\\\\n')
        file.writelines(footer)

        # for f in folders:
        #     print(f"{'=' * 80}\n")
        #     print(f"\tResults from folder: {f}\n")

        #     compile_results(f"../output/{models_to_compare[0]}/{f}/cam_metrics.json",
        #                 f"../output/{models_to_compare[1]}/{f}/cam_metrics.json")
        #     print(f"\n{'=' * 80}")
