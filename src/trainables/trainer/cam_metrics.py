import json
import os
import torch
import numpy as np
import torch.nn.functional as F
import warnings
import cv2
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import Normalizer

warnings.filterwarnings("ignore", category=UserWarning)


# SAUCE: https://openaccess.thecvf.com/content/CVPR2021W/RCV/papers/Poppi_Revisiting_the_Evaluation_of_Class_Activation_Mapping_for_Explainability_A_CVPRW_2021_paper.pdf

def get_grad_cam(model, class_to_backprop: int = 0, img_name=None, img=None):
    device = "cpu"

    if img is None:
        img = Image.open(img_name).convert("RGB")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

    # Transformações da imagem de entrada
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Pré-processando a imagem e transformando ela em um "mini-batch"
    img = preprocess(img)
    input_batch = img.unsqueeze(0)
    input_batch = input_batch.to(device)

    model = model.to(device)

    if model.__class__.__name__ == "UniFormer_Light" or model.__class__.__name__ == "UniFormer_Light_ABN_CF_GAP":
        model.train()
    else:
        model.eval()

    # Obtendo a classificação do modelo e calculando o gradiente da maior classe
    if model.__class__.__name__ == "ResNet":
        outputs = model(input_batch)[1]
    elif model.__class__.__name__ == "UniFormer_Light" or model.__class__.__name__ == "UniFormer_Light_ABN_CF_GAP":
        outputs = model(input_batch)[0]
    else:
        outputs = model(input_batch)

    prob = F.softmax(outputs).detach().cpu().numpy()

    prob1 = prob[0][class_to_backprop]

    print("Model classification: {} | Expected: {}".format(prob.argmax(), class_to_backprop))
    outputs[:, class_to_backprop].backward()

    # Obtendo informações dos gradienttes e construindo o "heatmap"

    gradients = model.get_activations_gradient()
    gradients = torch.mean(gradients, dim=[0, 2, 3])
    layer_output = model.get_activations(input_batch)

    for i in range(len(gradients)):
        layer_output[:, i, :, :] *= gradients[i]

    layer_output = layer_output[0, :, :, :]

    # Salvando imagens
    img = cv2.imread(img_name)
    img = np.float32(img)

    heatmap = torch.mean(layer_output, dim=0).detach().numpy()
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            for c in range(3):
                img[i][j][c] *= heatmap[i][j]

    img = np.uint8(img)

    return heatmap, img, prob1


def get_cam_metrics(model, identifier, dataset_name, imgs_dir, repeatet_holdout_index: int = -1):
    metrics_json = {}
    output_folder =  os.path.join('..', 'output', identifier, dataset_name)

    try:
        os.mkdir(os.path.join(output_folder, 'cams'))
    except OSError as error:
        print("CAM Metrics - The folder already exists???")

    imgs_array = os.listdir(imgs_dir)

    for img in imgs_array:

        c = int(img.split("/")[-1].split("_")[0])

        h1, i1, p1 = get_grad_cam(model, c, imgs_dir + "/" + img, None)
        h2, i2, p2 = get_grad_cam(model, c, imgs_dir + "/" + img, i1)

        denominator = np.cov(h2, h1)
        numerator_1 = np.std(h2)
        numerator_2 = np.std(h1)

        matrix = denominator / numerator_1 * numerator_2
        normalized_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

        m1 = np.mean(normalized_matrix)

        if np.isnan(m1):
            m1 = 0.00001

        m2 = Normalizer(norm='l1').transform(h1)
        m2 = np.mean(m2)
        #print(m2)

        #m3 = (max(0, p1-p2)/p1)*100
        m3 = (max(0, p1 - p2) / p1)

        adcc = 3 * ((1 / m1) + (1 / 1 - m2) + (1 / 1 - m3)) ** -1

        print(
            "Coherency (HB): {} | Complexity (LB): {} | Average Drop (LB): {} | ADCC (HB): {}".format(m1, m2, m3, adcc))
        img_name = img.split('/')[-1]
        metrics_json[img_name] = {"coherency": float(m1), "complexity": float(m2), "average_drop": float(m3),
                                  "adcc": float(adcc)}

        #cv2.imwrite(os.path.join(output_folder, "cams", f"i1_{img_name}.png") if repeatet_holdout_index == -1 else os.path.join(output_folder, "cams", f"i1_{img_name}_{repeatet_holdout_index}.png"), i1)
        cv2.imwrite(os.path.join(output_folder, "cams", f"i1_{img_name}.png"), i1)
        #cv2.imwrite(f"{output_folder}/cams/i2_{img_name}.png" if repeatet_holdout_index == -1 else f"{output_folder}/cams/i1_{img_name}_{repeatet_holdout_index}.png", i2)

    with open(os.path.join(output_folder, "cam_metrics.json") if repeatet_holdout_index == -1 else os.path.join(output_folder, f"cam_metrics_{repeatet_holdout_index}.json"), "w") as f:
        json.dump(metrics_json, f)
