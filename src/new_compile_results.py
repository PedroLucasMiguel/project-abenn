import os
import json
import numpy as np

toMeanOut = np.zeros(shape=(5, 4))

def compile_result_to_table(model_name, dt_name, file, toMeanOutIndex) -> None:
    toMean = np.zeros(shape=(5, 4))

    for i in range(5):
      f = open(os.path.join('..', 'output', model_name, dt_name, f'cam_metrics_{i}.json'), "r")
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
      sumns = np.nan_to_num(sumns, nan=0.0)
      toMean[i][0] += round(((0.0 if np.isnan(sumns[0][0]) else sumns[0][0] / n_samples)*100), 2)
      toMean[i][1] += round(((0.0 if np.isnan(sumns[0][0]) else sumns[1][0] / n_samples)*100), 2)
      toMean[i][2] += round(((0.0 if np.isnan(sumns[0][0]) else sumns[2][0] / n_samples)*100), 2)
      toMean[i][3] += round(((0.0 if np.isnan(sumns[0][0]) else sumns[3][0] / n_samples)*100), 2)

    metrics = np.mean(toMean, axis=0)
    toMeanOut[toMeanOutIndex][0] = metrics[0]
    toMeanOut[toMeanOutIndex][1] = metrics[1]
    toMeanOut[toMeanOutIndex][2] = metrics[2]
    toMeanOut[toMeanOutIndex][3] = metrics[3]
    file.write(f'{dt_name.split('_')[0]} & {round(metrics[0], 2)} & {round(metrics[1], 2)} & {round(metrics[2], 2)} & {round(metrics[3], 2)} \\\\\n')


def compile_result_to_table_validation(model_name, dt_name, file, toMeanOutIndex) -> None:
    toMean = np.zeros(shape=(5, 4))

    for i in range(5):
      f = open(os.path.join('..', 'output', model_name, dt_name, f'training_results_{i}.json'), "r")
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
      sumns = np.nan_to_num(sumns, nan=0.0)
      toMean[i][0] += round(((0.0 if np.isnan(sumns[0][0]) else sumns[0][0] / n_samples)*100), 2)
      toMean[i][1] += round(((0.0 if np.isnan(sumns[0][0]) else sumns[1][0] / n_samples)*100), 2)
      toMean[i][2] += round(((0.0 if np.isnan(sumns[0][0]) else sumns[2][0] / n_samples)*100), 2)
      toMean[i][3] += round(((0.0 if np.isnan(sumns[0][0]) else sumns[3][0] / n_samples)*100), 2)

    metrics = np.mean(toMean, axis=0)
    toMeanOut[toMeanOutIndex][0] = metrics[0]
    toMeanOut[toMeanOutIndex][1] = metrics[1]
    toMeanOut[toMeanOutIndex][2] = metrics[2]
    toMeanOut[toMeanOutIndex][3] = metrics[3]
    file.write(f'{dt_name.split('_')[0]} & {round(metrics[0], 2)} & {round(metrics[3], 2)} \\\\\n')


def compile_result_to_table_test(model_name, dt_name, file, toMeanOutIndex) -> None:
    toMean = np.zeros(shape=(5, 4))

    for i in range(5):
      f = open(os.path.join('..', 'output', model_name, dt_name, f'test_results_{i}.json'), "r")
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
      sumns = np.nan_to_num(sumns, nan=0.0)
      toMean[i][0] += round(((0.0 if np.isnan(sumns[0][0]) else sumns[0][0] / n_samples)*100), 2)
      toMean[i][1] += round(((0.0 if np.isnan(sumns[0][0]) else sumns[1][0] / n_samples)*100), 2)
      toMean[i][2] += round(((0.0 if np.isnan(sumns[0][0]) else sumns[2][0] / n_samples)*100), 2)
      toMean[i][3] += round(((0.0 if np.isnan(sumns[0][0]) else sumns[3][0] / n_samples)*100), 2)

    metrics = np.mean(toMean, axis=0)
    toMeanOut[toMeanOutIndex][0] = metrics[0]
    toMeanOut[toMeanOutIndex][1] = metrics[1]
    toMeanOut[toMeanOutIndex][2] = metrics[2]
    toMeanOut[toMeanOutIndex][3] = metrics[3]
    file.write(f'{dt_name.split('_')[0]} & {round(metrics[0], 2)} & {round(metrics[3], 2)} \\\\\n')

if __name__ == "__main__":
    models = [
        #'RESNET50',
        #'RESNET50_ABN_CF_GAP',
        #'DENSENET201',
        #'DENSENET201_ABN_CF_GAP',
        #'CONVNEXT_SMALL',
        #'CONVNEXT_ABN_CF_GAP',
        #'EFFICIENTNET',
        'EFFICIENTNET_ABN_CF_GAP',
        #'COATNETB0',
        #'COATNET_ABN_CF_GAP',
        #'RESNEXT50',
        #'RESNEXT50_ABN_CF_GAP',
        #'UNIFORMER_BASELINE',
        #'UNIFORMER_ABN_CF_GAP',
    ]

    with open( os.path.join('..', 'tables', f'table_metrics.txt'), 'w') as file:
      for model_to_compare in models:
        print(model_to_compare)
        folders = os.listdir(os.path.join('..', 'output', model_to_compare))

        i = 0
    
        headers = [
            '\\begin{table}[H]\n',
            f'\\caption{{Percentage values of coherence (CO), complexity (COM), drop in confidence (CD) and ADCC for the {model_to_compare} model, considering all datasets.}}\n',
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
        file.writelines(headers)
        for f in folders:
            compile_result_to_table(model_to_compare, f, file, i)
            i += 1
        file.write('\\midrule\n')
        means = np.mean(toMeanOut, axis=0)
        file.write(f'\\textbf{{Mean}} & {round(means[0], 2)} & {round(means[1], 2)} & {round(means[2], 2)} & {round(means[3], 2)} \\\\\n')
        file.writelines(footer)
        file.writelines('\n\n')

    toMeanOut = np.zeros(shape=(5, 4))

    with open( os.path.join('..', 'tables', f'table_training.txt'), 'w') as file:
      for model_to_compare in models:
        print(model_to_compare)
        folders = os.listdir(os.path.join('..', 'output', model_to_compare))

        i = 0
    
        headers = [
            '\\begin{table}[H]\n',
            f'\\caption{{Percentage values of accuracy (ACC) and F1 for the {model_to_compare} model, considering all datasets (Validation and repeated holdout).}}\n',
            '\\begin{tabularx}{\\textwidth}{CCC}\n',
            '\\toprule',
            '\\textbf{Dataset} & ACC$\\uparrow$ & F1$\\uparrow$ \\\\',
            '\\midrule\n'
        ]
        footer = [
            #'\\midrule\n',
            #'\\textbf{Mean} & X & X & X & X \\\\\n',
            '\\bottomrule\n',
            '\\end{tabularx}\n',
            '\\end{table}'
        ]
        file.writelines(headers)
        for f in folders:
            compile_result_to_table_validation(model_to_compare, f, file, i)
            i += 1
        file.write('\\midrule\n')
        means = np.mean(toMeanOut, axis=0)
        file.write(f'\\textbf{{Mean}} & {round(means[0], 2)} & {round(means[1], 2)} & {round(means[2], 2)} & {round(means[3], 2)} \\\\\n')
        file.writelines(footer)
        file.writelines('\n\n')

    toMeanOut = np.zeros(shape=(5, 4))

    with open( os.path.join('..', 'tables', f'table_test.txt'), 'w') as file:
      for model_to_compare in models:
        print(model_to_compare)
        folders = os.listdir(os.path.join('..', 'output', model_to_compare))

        i = 0
    
        headers = [
            '\\begin{table}[H]\n',
            f'\\caption{{Percentage values of accuracy (ACC) and F1 for the {model_to_compare} model, considering all datasets (Tests and repeated holdout).}}\n',
            '\\begin{tabularx}{\\textwidth}{CCC}\n',
            '\\toprule',
            '\\textbf{Dataset} & ACC$\\uparrow$ & F1$\\uparrow$ \\\\',
            '\\midrule\n'
        ]
        footer = [
            #'\\midrule\n',
            #'\\textbf{Mean} & X & X & X & X \\\\\n',
            '\\bottomrule\n',
            '\\end{tabularx}\n',
            '\\end{table}'
        ]
        file.writelines(headers)
        for f in folders:
            compile_result_to_table_test(model_to_compare, f, file, i)
            i += 1
        file.write('\\midrule\n')
        means = np.mean(toMeanOut, axis=0)
        file.write(f'\\textbf{{Mean}} & {round(means[0], 2)} & {round(means[1], 2)} & {round(means[2], 2)} & {round(means[3], 2)} \\\\\n')
        file.writelines(footer)
        file.writelines('\n\n')