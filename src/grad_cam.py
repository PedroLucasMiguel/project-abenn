import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import warnings
import cv2
from PIL import Image
from torchvision import transforms
from model.densenet import DenseNet201ABENN
warnings.filterwarnings("ignore", category=UserWarning) 

IMG_NAME = "../samples/han.png"

device = "cpu"
print(f"Using {device}")

img = Image.open(IMG_NAME).convert("RGB")

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

# Construindo e carregando o treinamento do modelo
baseline_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
model = DenseNet201ABENN(baseline_model, 2)
model.load_state_dict(torch.load("c/fbest_model_3_f1=1.0000.pt"))
model = model.to(device)

model.eval()

# Obtendo a classificação do modelo e calculando o gradiente da maior classe
outputs = model(input_batch)
print(outputs[0])
class_to_backprop = F.softmax(outputs[0]).detach().cpu().numpy().argmax()
print(class_to_backprop)
print("\nClassificação do modelo: {}".format(class_to_backprop))
outputs[0][:, class_to_backprop].backward()

# Obtendo informações dos gradienttes e construindo o "heatmap"
gradients = model.get_activations_gradient()
gradients = torch.mean(gradients, dim=[0, 2, 3])
layer_output = model.get_activations(input_batch)

for i in range(len(gradients)):
    layer_output[:, i, :, :] *= gradients[i]

layer_output = layer_output[0, : , : , :]

# Salvando imagens
img = cv2.imread(IMG_NAME)

heatmap = torch.mean(layer_output, dim=0).detach().numpy()
heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) 
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.6 + img
cv2.imwrite("../output/gradient.jpg", heatmap)
final_img = np.concatenate((img, superimposed_img), axis=1)
cv2.imwrite("../output/map.jpg", final_img)

print("Imagens salvas em output/gradient.jpg e output/map.jpg\n")