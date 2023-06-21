import json
import torch
import os
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cupy as cp
from torchvision import transforms
from torchvision import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.densenet import DenseNet201ABENN

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

K_FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.1
MOMENTUM = 0.9

'''
EPOCHS = 90
BATCH_SIZE = 256
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9
'''

'''
EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
'''

N_CLASSES = 2

# Definindo as transformações que precisam ser feitas no conjunto de imagens
preprocess = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Carregando o dataset a partir da pasta
dataset_train = datasets.PCAM(root="../pcam_dataset", split="train", transform=preprocess, download=True)
dataset_validation = datasets.PCAM(root="../pcam_dataset", split="val", transform=preprocess, download=True)

# Criando o dataset com split 80/20 (Perfeitamente balanceado)
#dataset_train, dataset_validation = random_split(dataset, [0.8, 0.2])

# Criando os "loaders" para o nosso conjunto de treino e validação
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, pin_memory=True)
testloader = torch.utils.data.DataLoader(dataset_validation, batch_size=BATCH_SIZE, pin_memory=True)

# Utiliza GPU caso possível
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for training")

# Cria o modelo e define o dispositivo de execução
baseline = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
model = DenseNet201ABENN(baseline, N_CLASSES)
model.to(device)
model = nn.DataParallel(model)

# Definindo nossa função para o calculo de loss e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=LEARNING_RATE)
#optimizer = optim.SGD(model.parameters(), momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, lr=LEARNING_RATE)
#optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=(30, 60), gamma=0.1)

# Melhor resultado da métrica F1 (utilizado no processo de checkpoint)
best_f1 = 0
best_f1_file_name = ""

metrics_json = {}


# Iniciando o processo de treinamento
for epoch in range(0, EPOCHS):
    print(f"Doing for epoch - {epoch+1}")
    print("Training...")
    model.train()
    
    for i, data in enumerate(trainloader, 0):
        
        img, label = data
        img = img.to(device)
        label = label.to(device)
        pred = model(img)
        # pred[0] = previsão | pred[1]= model.att (estou fazendo isso por conta do nn.DataParallel)
        loss = criterion(pred[0], label)

        att = pred[1].detach()
        #att = model.att.detach()
        att = cp.asarray(att)
        cam_normalized = cp.zeros((att.shape[0], att.shape[2], att.shape[3]))

        for i in range(att.shape[0]):
            s = cp.sum(att[i,0,:,:])
            cam_normalized[i,:,:] = cp.divide(att[i,0,:,:], s)

        # Realizando a média dos batches
        m = cp.mean(cam_normalized, axis=0)

        ce = 10*cp.sum(m*cp.log(m))

        loss = loss - ce.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #scheduler.step()
        
    accuracy = []
    precision = []
    recall = []
    f1 = []

    print("Validating...")
    # Iniciando o processo de validação
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs[0].data, 1)
            # Convertendo os arrays das labels e das previsões para uso em CPU
            label_cpu = label.cpu()
            predicted_cpu = predicted.cpu()
            # Calculando as métricas
            precision.append(precision_score(label_cpu, predicted_cpu, average="binary"))
            recall.append(recall_score(label_cpu, predicted_cpu, average="binary"))
            f1.append(f1_score(label_cpu, predicted_cpu, average="binary"))
            accuracy.append(accuracy_score(label_cpu, predicted_cpu))

    # Apresentando as métricas
    accuracy = np.mean(accuracy)
    precision = np.mean(precision)
    recall = np.mean(recall)
    f1 = np.mean(f1)
    print("\nResults:")
    print(f"Accuracy for Epoch {epoch+1}: {100.0 * accuracy}%")
    print(f"Precision {(precision)}")
    print(f"Recall {recall}")
    print(f"F1 {f1}")
        
    # Se o resultado possuir a melhor medida F de todo o processo, salve o treinamento
    if f1 > best_f1:
        print(f"\nSaving saving training for Epoch={epoch+1}")
        best_f1 = f1
        torch.save(model.state_dict(), f"checkpoints/e_{epoch+1}_pcam_savestate.pth")

        if best_f1_file_name != "":
            os.remove("checkpoints/{}".format(best_f1_file_name))

        best_f1_file_name = f"e_{epoch+1}_pcam_savestate.pth"

    metrics_json[epoch+1] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }
    print('--------------------------------')

# Exporta as métricas em um arquivo .json
with open("../output/metrics2.json", "w") as json_file:
    json.dump(metrics_json, json_file)