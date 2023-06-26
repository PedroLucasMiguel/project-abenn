import torch
import torch.nn as nn
import numpy as np
import cv2
import warnings
import torch.nn.functional as F
import os
from torchvision import transforms
from lime import lime_image
from PIL import Image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
from model.densenet import DenseNet201ABENN
warnings.filterwarnings("ignore", category=UserWarning) 

# Função responsável por obter a explicação LIME de uma determinada imagem
# Criando a instância do Lime
explainer = lime_image.LimeImageExplainer()

IMG_NAME = "../samples/poke.png"

# Definindo processos de pré-processamento
def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((224, 224)),
    ])    

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]) 

    return transf

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

device = "cuda"
print(f"Usando {device}")

# Função responsável por ler a imagem em um formato que o LIME consegue trabalhar
def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 
            
img = get_image(IMG_NAME)

# Construindo e carregando o treinamento do modelo
baseline_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
model = DenseNet201ABENN(baseline_model, 2)
model.load_state_dict(torch.load("checkpoint_cat_dog/best_model_10_f1=0.9910.pt"))
model = model.to(device)

# Função de "classificação"
# É necessária para o LIME
def classify_func(img):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in img), dim=0)
    batch = batch.to(device)
    outputs = model(batch)

    return F.softmax(outputs[0], dim=1).detach().cpu().numpy()

# Criando a imagem com a explicação e apresentando ela em um plot
explanation = explainer.explain_instance(np.array(pill_transf(img)), classify_func, top_labels=1, hide_color=0, num_samples=2000)
print("\nClassificação do modelo: {}\n".format(explanation.top_labels[0]))
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
img_boundry1 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry1)
plt.savefig("../output/lime.jpg")