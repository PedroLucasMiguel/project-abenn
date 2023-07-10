import os
import torch
import json
import shutil
import numpy as np
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
import cupy as cp
import cv2
from torchvision import transforms
from torchvision import datasets
from model.densenet import DenseNet201ABENN
from model.densenet_baseline import DenseNetGradCam
from cam_metrics import get_cam_metrics

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9

N_CLASSES = 2

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_augment_dataset(dataset_dir:str, repeats:int):

    # Criando estrutura para o novo dataset
    classes = os.listdir(dataset_dir)
    classes.sort()

    augmented_dataset_path = dataset_dir+"_aug"

    try:
        os.mkdir(augmented_dataset_path)
        for c in classes:
            os.mkdir(f"{augmented_dataset_path}/{c}/")

        # Copiando as imagens originais para o novo diretório
        for c in classes:
            imgs = os.listdir(f"{dataset_dir}/{c}")

            for img in imgs:
                shutil.copyfile(f"{dataset_dir}/{c}/{img}", f"{augmented_dataset_path}/{c}/{img}")
        
        j = 0
        
        for i in range(repeats):
            # Criando as novas imagens
            preprocess = transforms.Compose([
                transforms.ColorJitter(hue=0.05, saturation=0.05),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
            ])

            dataset = datasets.ImageFolder(augmented_dataset_path, preprocess)
            
            for i, c in dataset:
                cv2.imwrite(f"{augmented_dataset_path}/{classes[c]}/{j}.png", np.asarray(i))
                j += 1

        print("The Dataset was augmented succesfully")

    except OSError as error:
        print("The Dataset is already augmented")

    return augmented_dataset_path

def get_dataset(dataset_dir:str):

    # TODO - Separar 15% do dataset, de preferencia de forma "balanceada" a fim de que nós possamos realizar a obtenção das métricas
    # de qualidade de um CAM. O uso das imagens "augmentadas" não parece colaborar...

    # Definindo as transformações que precisam ser feitas no conjunto de imagens
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Carregando o dataset a partir da pasta
    dataset = datasets.ImageFolder(dataset_dir, preprocess)

    return dataset

def get_dataflow(dataset):

    train, val = random_split(dataset, [0.7, 0.3])
    val, test = random_split(val, [0.5, 0.5])

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=1,
        shuffle=True,
    )

    return train_loader, val_loader, test_loader

def get_optimizer_scheduler(model):
    optimizer = optim.Adamax(model.parameters(), lr=LEARNING_RATE)
    return optimizer

def get_criterion():
    return nn.CrossEntropyLoss().to(device)

def procedure(model,
              loaders, 
              output_folder_name:str, 
              use_custom_train_step = None, 
              use_custom_validation_step = None,
              dataset_dir:str = "../datasets/UCSB"):
    
    
    optimizer = get_optimizer_scheduler(model)
    criterion = get_criterion()

    train_loader, val_loader, test_loader = loaders

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
    
    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch[0].to(device), batch[1].to(device)
            y_pred = model(x)
            return y_pred, y

    final_json = {}

    # Pytorch-ignite bit
    val_metrics = {
        "accuracy": Accuracy(),
        "precision": Precision(average='macro'),
        "recall": Recall(average='macro'),
        "f1": (Precision(average=False) * Recall(average=False) * 2 / (Precision(average=False) + Recall(average=False))).mean(),
        "loss": Loss(criterion)
    }

    if use_custom_train_step:
        trainer = Engine(train_step)
    else:
        trainer = create_supervised_trainer(model, optimizer, criterion, device)

    if use_custom_validation_step:
        val_evaluator = Engine(validation_step)
    else:
        val_evaluator = create_supervised_evaluator(model, val_metrics, device)

    for name, metric in val_metrics.items():
        metric.attach(val_evaluator, name)

    train_bar = ProgressBar(desc="Training...")
    val_bar = ProgressBar(desc="Evaluating...")
    train_bar.attach(trainer)
    val_bar.attach(val_evaluator)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics

        final_json[trainer.state.epoch] = metrics

        print(f"Validation Results - Epoch[{trainer.state.epoch}] {final_json[trainer.state.epoch]}")

    def score_function(engine):
        return engine.state.metrics["f1"]

    model_checkpoint = ModelCheckpoint(
        f"../output/{output_folder_name}_TRAIN",
        require_empty=False,
        n_saved=1,
        filename_prefix=f"{dataset_dir.split('/')[-1]}",
        score_function=score_function,
        score_name="f1",
        global_step_transform=global_step_from_engine(trainer),
    )
    
    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    trainer.run(train_loader, max_epochs=EPOCHS)

    with open(f"../output/{output_folder_name}_TRAIN/training_{dataset_dir.split('/')[-1]}.json", "w") as f:
        json.dump(final_json, f)

    model.load_state_dict(torch.load(model_checkpoint.last_checkpoint))

    get_cam_metrics(model, output_folder_name, test_loader.sampler.data_source.dataset.dataset.imgs)


if __name__ == "__main__":

    baseline = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    model1 = DenseNet201ABENN(baseline, N_CLASSES)
    model1 = model1.to(device)

    model2 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    model2.classifier = nn.Linear(in_features=1920, out_features=2, bias=True)
    model2 = DenseNetGradCam(model2)
    model2.to(device)

    dataset = get_dataset(get_augment_dataset("../datasets/UCSB", repeats=2))
    loaders = get_dataflow(dataset)

    procedure(model1, loaders, "ABN", True, True)
    procedure(model2, loaders, "DENSE", False, False)