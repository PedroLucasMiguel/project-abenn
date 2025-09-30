import os
import json
import argparse
from decimal import Decimal

def find_best_epoch(json_path, verbose=False):
    """
    Lê o arquivo JSON de resultados de treinamento usando Decimal para precisão exata,
    retorna as métricas da epoch com maior F1 (primeira ocorrência).
    Se verbose=True, imprime todas as F1 lidas para debug.
    """
    # Carrega JSON com Decimal para floats
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f, parse_float=Decimal)

    best_epoch = None
    best_metrics = None
    best_f1 = Decimal('-Infinity')

    for epoch_str, metrics in data.items():
        f1 = metrics.get('f1')
        if not isinstance(f1, Decimal):
            # ignora se não for número
            if verbose:
                print(f"Ignorando epoch {epoch_str}: f1 não é numérico -> {f1}")
            continue
        if verbose:
            print(f"Debug: dataset epoch {epoch_str} -> f1 = {f1}")
        # Considera primeira ocorrência em caso de empate
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch_str
            best_metrics = metrics

    return best_epoch, best_metrics



def main():
    parser = argparse.ArgumentParser(
        description='Para cada dataset de um modelo, encontra a epoch com maior F1 e exibe suas métricas.'
    )

    MODEL_NAME = 'RESNET50'

    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Caminho para a pasta base que contém os resultados (padrão: output)'
    )
    args = parser.parse_args()

    model_dir = os.path.join(args.output_dir, MODEL_NAME)
    if not os.path.isdir(model_dir):
        print(f"Erro: diretório do modelo não encontrado: {model_dir}")
        return

    print(f"Melhores métricas (maior F1) por dataset para o modelo: {MODEL_NAME}\n")

    found_any = False
    for dataset in sorted(os.listdir(model_dir)):
        dataset_dir = os.path.join(model_dir, dataset)
        json_path = os.path.join(dataset_dir, 'training_results.json')
        if not os.path.isfile(json_path):
            print(f"Ignorado {dataset}: arquivo training_results.json não encontrado.")
            continue

        found_any = True
        epoch, metrics = find_best_epoch(json_path)
        if epoch is None:
            print(f"Dataset {dataset}: nenhuma métrica F1 encontrada.")
            continue

        print(f"Dataset: {dataset}")
        print(f"  Melhor epoch: {epoch} (F1 = {metrics.get('f1'):.4f})")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        print()

    if not found_any:
        print(f"Nenhum dataset válido com training_results.json em {model_dir}.")

if __name__ == '__main__':
    main()