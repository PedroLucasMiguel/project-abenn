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
from model.densenet import DenseNet201ABENN

from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall, Loss
from ignite.contrib.metrics.roc_auc import ROC_AUC
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

N_CLASSES = 2

# Definindo as transformações que precisam ser feitas no conjunto de imagens
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Carregando o dataset a partir da pasta
dataset = datasets.ImageFolder("../dataset/", preprocess)

# Criando o dataset com split 80/20 (Perfeitamente balanceado)
dataset_train, dataset_validation = random_split(dataset, [0.8, 0.2])

# Criando os "loaders" para o nosso conjunto de treino e validação
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, pin_memory=True)
val_loader = torch.utils.data.DataLoader(dataset_validation, batch_size=BATCH_SIZE, pin_memory=True)

# Utiliza GPU caso possível
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for training")

# Cria o modelo e define o dispositivo de execução
baseline = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
model = DenseNet201ABENN(baseline, N_CLASSES)
model.to(device)

# Criterion e optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

# Pytorch-ignite bit
val_metrics = {
    "accuracy": Accuracy(),
    "precision": Precision(),
    "recall": Recall(),
    "loss": Loss(criterion)
}

def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch[0].to(device), batch[1].to(device)
    y_pred = model(x)
    loss = criterion(y_pred, y)

    att = model.att.detach()
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
    optimizer.step()

    return loss.item()


trainer = Engine(train_step)
val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
train_bar = ProgressBar(desc="Training...")
val_bar = ProgressBar(desc="Evaluating traing...")
train_bar.attach(trainer)
val_bar.attach(val_evaluator)

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    print(f"Validation Results - Epoch[{trainer.state.epoch}] {metrics}")


def score_function(engine):
    return engine.state.metrics["accuracy"]


model_checkpoint = ModelCheckpoint(
    "checkpoint",
    require_empty=False,
    n_saved=1,
    filename_prefix="best",
    score_function=score_function,
    score_name="accuracy",
    global_step_transform=global_step_from_engine(trainer),
)
  
val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

trainer.run(train_loader, max_epochs=EPOCHS)