import os
import json
from decimal import Decimal


# Defina aqui os pares de modelos para comparação: [ ["baseline1", "proposed1"], ["baseline2","proposed2"], ... ]
model_pairs = [
    ['RESNET50_ABN_CF_GAP', 'RESNET50'],
    ['DENSENET201_ABN_CF_GAP', 'DENSENET201'],
    ['CONVNEXT_ABN_CF_GAP', 'CONVNEXT_SMALL'],
    ['EFFICIENTNET_ABN_CF_GAP', 'EFFICIENTNET',],
    ['COATNET_ABN_CF_GAP', 'COATNETB0'],
    ['RESNEXT50_ABN_CF_GAP', 'RESNEXT50'],
]

output_dir = 'output'  # pasta base onde estão os diretórios dos modelos
datasets = []  # opcional: lista de nomes de datasets; se vazio, detecta automaticamente
metrics_keys = ["accuracy", "precision", "recall", "f1", "loss"]

def find_best_metrics(json_path):
    """
    Lê JSON com Decimal e retorna um dict de métricas da epoch com maior F1.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f, parse_float=Decimal)

    best_f1 = Decimal('-Infinity')
    best_metrics = None
    for metrics in data.values():
        f1 = metrics.get('f1')
        if not isinstance(f1, Decimal):
            continue
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = metrics
    return best_metrics


def average_metrics_across_datasets(model_name):
    """
    Para cada dataset em output/<model_name> encontra a melhor epoch por F1,
    calcula a média das métricas entre todos os datasets válidos.
    Retorna dict com médias por métrica.
    """
    model_dir = os.path.join(output_dir, model_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Modelo não encontrado: {model_dir}")

    ds_list = datasets or sorted(
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
    )

    sums = {k: Decimal('0') for k in metrics_keys}
    count = 0
    for ds in ds_list:
        json_path = os.path.join(model_dir, ds, 'training_results.json')
        if not os.path.isfile(json_path):
            continue
        best = find_best_metrics(json_path)
        if best is None:
            continue
        for k in metrics_keys:
            val = best.get(k)
            if isinstance(val, Decimal):
                sums[k] += val
        count += 1
    if count == 0:
        return {k: None for k in metrics_keys}

    return {k: (sums[k] / count) for k in metrics_keys}


# Calcula médias para cada modelo em cada par
table_rows = []
for baseline, proposed in model_pairs:
    base_avgs = average_metrics_across_datasets(baseline)
    prop_avgs = average_metrics_across_datasets(proposed)
    table_rows.append((f"{baseline} vs {proposed}", base_avgs, prop_avgs))

# Monta tabela LaTeX
num_metrics = len(metrics_keys)
col_fmt = 'l' + 'cc' * num_metrics
header = ' & '.join(
    f"\\multicolumn{{2}}{{c}}{{{key.upper()}}}" for key in metrics_keys
)
subhdr = ' & '.join('Baseline & Proposed' for _ in metrics_keys)

latex = []
latex.append(r'\begin{table}[H]')
latex.append(r'\centering')
latex.append(f"\\begin{{tabular}}{{{col_fmt}}}")
latex.append(r'\toprule')
latex.append(f"Model Pair & {header} \\")
latex.append(f" & {subhdr} \\")
latex.append(r'\midrule')
for label, base, prop in table_rows:
    vals = []
    for k in metrics_keys:
        b = f"{base[k]}" if base[k] is not None else '-'
        p = f"{prop[k]}" if prop[k] is not None else '-'
        vals.extend([b, p])
    latex.append(f"{label} & {' & '.join(vals)} \\")
latex.append(r'\bottomrule')
latex.append(r'\end{tabular}')
latex.append(r'\caption{Médias das métricas (accuracy, precision, recall, F1, loss) obtidas pela melhor epoch de cada dataset, comparando baseline e proposto.}')
latex.append(r'\label{tab:comparison_results}')
latex.append(r'\end{table}')

# Imprime tabela no console
print("\n".join(latex))

