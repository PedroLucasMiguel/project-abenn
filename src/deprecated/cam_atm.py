import cv2
import json
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import warnings
from PIL import Image
from model.resnet_abn_cf import resnet50_cf
from torchvision import transforms
from matplotlib import pyplot as plt
from tv_framework import *
warnings.filterwarnings("ignore", category=UserWarning) 

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
    if model.__class__.__name__ == "ResNet":
        outputs = model(input_batch)[1]
    else:
        outputs = model(input_batch)

    prob = F.softmax(outputs).detach().cpu().numpy()

    prob1 = prob[0][0]

    print("Model classification: {} | Expected: {}".format(prob.argmax(), class_to_backprop))
    outputs[:, 0].backward()

    # Obtendo informações dos gradienttes e construindo o "heatmap"
    
    gradients = model.get_att_gradient().detach().numpy()[0,0,:,:]
    layer_output = model.att.detach().numpy()[0,0,:,:]
    heatmap = np.dot(gradients, layer_output)
    
    img = cv2.imread(img_name)
    img = np.float32(img)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            for c in range(3):
                img[i][j][c] *= heatmap[i][j]

    img = np.uint8(img)

    cv2.imwrite("teste.png", img)

    '''
    gradients = model.get_activations_gradient()

    gradients = torch.mean(gradients, dim=[0, 2, 3])
    layer_output = model.get_activations(input_batch)

    for i in range(len(gradients)):
        layer_output[:, i, :, :] *= gradients[i]

    layer_output = layer_output[0, : , : , :]
    

    heatmap = model.att.detach().numpy()

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
    '''

    #return heatmap, img, prob1
    return

EPOCHS = 10

def train_resnet50_abn_cf(dn:str, use_gpu_n:int = 0) -> None:
    n_classes = len(os.listdir(f"../datasets/{dn}"))
    model = resnet50_cf(True)
    model.fc = nn.Linear(512 * model.block.expansion, n_classes)

    cpf = ComparatorFramewok(epochs=EPOCHS, 
                               model=model, 
                               use_gpu_n=use_gpu_n, 
                               dataset_name=dn, 
                               use_augmentation=False,
                               batch_size=32)
    
    def train_step(engine, batch):
        cpf.model.train()
        cpf.optimizer.zero_grad()
        x, y = batch[0].to(cpf.device), batch[1].to(cpf.device)
        y_pred = cpf.model(x)
        loss = cpf.criterion(y_pred, y)

        att = cpf.model.att.detach()
        att = cp.asarray(att)
        cam_normalized = cp.zeros((att.shape[0], att.shape[2], att.shape[3]))

        for i in range(att.shape[0]):
            s = cp.sum(att[i,0,:,:])
            cam_normalized[i,:,:] = cp.divide(att[i,0,:,:], s)

        # Realizando a média dos batches
        m = cp.mean(cam_normalized, axis=0)

        ce = 10*cp.sum(m*cp.log(m))

        loss = loss - ce.item()
                
        loss.backward()
        cpf.optimizer.step()

        return loss.item()
            
    def validation_step(engine, batch):
        cpf.model.eval()
        with torch.no_grad():
            x, y = batch[0].to(cpf.device), batch[1].to(cpf.device)
            y_pred = cpf.model(x)
            return y_pred, y


    cpf.set_custom_train_step(train_step)
    cpf.set_custom_val_step(validation_step)
        
    cpf.procedure("RESNET_ABN_CF")

if __name__ == "__main__":
    #train_resnet50_abn_cf('PetImages')
    model = resnet50_cf(True)
    model.fc = nn.Linear(512 * model.block.expansion, 2)
    model.load_state_dict(torch.load('../output/RESNET_ABN_CF/PetImages_2_CV/train_model_1_f1=0.9963.pt'))
    get_grad_cam(model, 0, '../datasets/PetImages_2_CV/test/0_2.png')
    