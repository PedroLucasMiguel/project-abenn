import json
import torch
import os
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torchvision import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.densenet import DenseNet201ABENN
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

'''
K_FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.1
MOMENTUM = 0.9
'''

K_FOLDS = 5
EPOCHS = 90
BATCH_SIZE = 64
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

# Definindo as transformações que precisam ser feitas no conjunto de imagens
preprocess = transforms.Compose([
    transforms.Resize((224, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Carregando o dataset a partir da pasta
dataset = datasets.ImageFolder("../dataset2/", preprocess)

# Criando o dataset com split 80/20 (Perfeitamente balanceado)
dataset_train, dataset_validation = random_split(dataset, [0.8, 0.2])
#dataset_train, dataset_validation = random_split(dataset, [0.7, 0.3])
#dataset_test, dataset_validation = random_split(dataset_validation, [0.15, 0.15])

# Criando os "loaders" para o nosso conjunto de treino e validação
trainloader = torch.utils.data.DataLoader(
                dataset_train, 
                batch_size=BATCH_SIZE)
testloader = torch.utils.data.DataLoader(
                dataset_validation,
                batch_size=BATCH_SIZE)

# Utiliza GPU caso possível
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for training")

# Cria o modelo, reseta os pesos e define o dispositivo de execução
baseline = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
model = DenseNet201ABENN(baseline, 10)
#model = nn.DataParallel(model)
model.to(device)

# Definindo nossa função para o calculo de loss e o otimizador
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adamax(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.SGD(model.parameters(), momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 60], gamma=0.1)

# Melhor resultado da métrica F1 (utilizado no processo de checkpoint)
best_f1 = 0
best_f1_file_name = ""

metrics_json = {}

print(model)

# Iniciando o processo de treinamento
for epoch in range(0, EPOCHS):
    model.train()
    print(f"Doing for epoch - {epoch+1}")
        
    print("Training...")
    for i, data in enumerate(trainloader, 0):
        img, label = data
        img = img.to(device)
        label = label.to(device)
        model.train()
        pred = model(img)
        loss = criterion(pred, label)
        
        att = model.att.detach().cpu().numpy()
        cam_normalized = np.zeros((att.shape[0], att.shape[2], att.shape[3]))

        for i in range(att.shape[0]):
            s = np.sum(att[i,0,:,:])
            cam_normalized[i,:,:] = np.divide(att[i,0,:,:], s)

        # EU NÃO SEI O QUE EU FAÇO COM O BATCH, ENTÃO ESTOU FAZENDO A MÉDIA DE TODOS OS BATCHS, SOCORRO
        m = np.mean(cam_normalized, axis=0)

        ce = 10*np.sum(m*np.log(m))

        loss = loss - ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    scheduler.step()
        
    accuracy = []
    precision = []
    recall = []
    f1 = []
    model.eval()

    print("Validating...")
    # Iniciando o processo de validação
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            # Convertendo os arrays das labels e das previsões para uso em CPU
            label_cpu = label.cpu()
            predicted_cpu = predicted.cpu()
            # Calculando as métricas
            precision.append(precision_score(label_cpu, predicted_cpu, average="macro"))
            recall.append(recall_score(label_cpu, predicted_cpu, average="macro"))
            f1.append(f1_score(label_cpu, predicted_cpu, average="macro"))
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
        torch.save(model.state_dict(), f"checkpoints/e_{epoch+1}_savestate.pth")

        if best_f1_file_name != "":
            os.remove("checkpoints/{}".format(best_f1_file_name))

        best_f1_file_name = f"e_{epoch+1}_savestate.pth"

    metrics_json[epoch+1] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }
    print('--------------------------------')

# Exporta as métricas em um arquivo .json
with open("../output/metrics.json", "w") as json_file:
    json.dump(metrics_json, json_file)