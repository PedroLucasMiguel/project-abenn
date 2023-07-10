import json
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import warnings
import cv2
from PIL import Image
from torchvision import transforms
from model.densenet import DenseNet201ABENN
from matplotlib import pyplot as plt
from sklearn.preprocessing import Normalizer
warnings.filterwarnings("ignore", category=UserWarning) 

# SAUCE: https://openaccess.thecvf.com/content/CVPR2021W/RCV/papers/Poppi_Revisiting_the_Evaluation_of_Class_Activation_Mapping_for_Explainability_A_CVPRW_2021_paper.pdf

def get_grad_cam(model, class_to_backprop:int = 0, img_name = None, img = None):

    device = "cpu"

    if img is None:
        img = Image.open(img_name).convert("RGB")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

    # Transformações da imagem de entrada
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Pré-processando a imagem e transformando ela em um "mini-batch"
    img = preprocess(img)
    input_batch = img.unsqueeze(0)
    input_batch = input_batch.to(device)

    model = model.to(device)

    model.eval()

    # Obtendo a classificação do modelo e calculando o gradiente da maior classe
    outputs = model(input_batch)
    prob = F.softmax(outputs).detach().cpu().numpy()

    prob1 = prob[0][class_to_backprop]

    print("\nClassificação do modelo: {} | Esperado: {}".format(prob.argmax(), class_to_backprop))
    outputs[:, class_to_backprop].backward()

    # Obtendo informações dos gradienttes e construindo o "heatmap"
    gradients = model.get_activations_gradient()
    gradients = torch.mean(gradients, dim=[0, 2, 3])
    layer_output = model.get_activations(input_batch)

    for i in range(len(gradients)):
        layer_output[:, i, :, :] *= gradients[i]

    layer_output = layer_output[0, : , : , :]

    # Salvando imagens
    img = cv2.imread(img_name)
    img = np.float32(img)

    heatmap = torch.mean(layer_output, dim=0).detach().numpy()
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.float32(heatmap)

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            for c in range(3):
                img[i][j][c] *= heatmap[i][j]

    img = np.uint8(img)

    return heatmap, img, prob1

def get_cam_metrics(model, identifier, imgs_array):

    metrics_json = {}
    output_folder = f"../output/{identifier}"

    try:
        os.mkdir(f"../output/{identifier}")
    except OSError as error:
        print("a")

    for img, c in imgs_array:

        h1, i1, p1 = get_grad_cam(model, c, img, None)
        h2, i2, p2 = get_grad_cam(model, c, img, i1)

        de = np.cov(h2, h1)
        nu1 = np.std(h2)
        nu2 = np.std(h1)
        matrix = de/nu1*nu2
        n_matrix = (matrix + 1) / 2
        n_matrix = n_matrix / np.max(n_matrix)

        m1 = np.mean(n_matrix)

        m2_1 = Normalizer(norm='l1').fit(h1)
        m2 = m2_1.transform(h1)
        m2 = np.mean(m2)

        m3 = (max(0, p1-p2)/p1)*100

        adcc = 3*(((1/m1) + (1/1-m2) + (1/1-m3))**-1)

        print("M1: {} | M2: {} | M3: {} | ADCC: {}".format(m1,m2,m3,adcc))
        img_name = img.split('/')[-1]
        metrics_json[img_name] = {"m1": float(m1), "m2": float(m2), "m3": float(m3), "adcc": float(adcc)}

        #cv2.imwrite(f"{output_folder}/i1_{img_name}.png", i1)
        #cv2.imwrite(f"{output_folder}/i2_{img_name}.png", i2)

    with open(f"{output_folder}/{identifier}.json", "w") as f:
        json.dump(metrics_json, f)