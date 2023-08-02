import os
import torch
import json
import shutil
import numpy as np
import random
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

#EPOCHS = 10
#BATCH_SIZE = 8
#LEARNING_RATE = 0.001
#WEIGHT_DECAY = 0.0001
#MOMENTUM = 0.9

N_CLASSES = 2

class ComparatorFramewok:
    def __init__(self,
                 epochs:int = 10,
                 batch_size:int = 8,
                 lr:float = 0.001,
                 weight_decay:float = 0.0001,
                 momentum:float = 0.9,
                 model:nn.Module = None,
                 dataset_name:str = None,
                 use_augmentation:bool = True,
                 augmentation_repeats:int = 2) -> None:
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        self.optimizer = self.__get_optimizer()
        self.criterion = self.__get_criterion()
        self.__train_step = None
        self.__validation_step = None

        self.dataset_name = dataset_name

        self.use_augmentation = use_augmentation

        if self.use_augmentation:
            self.loaders = self.__get_loaders(
                                self.__get_dataset(
                                    self.__generate_augment_dataset(f"../datasets/{self.dataset_name}", augmentation_repeats)))
        else:
            self.loaders = self.__get_loaders(
                                self.__get_dataset(f"../datasets/{self.dataset_name}"))

        
        
        pass
    
    def set_custom_train_step(self, train_step):
        self.__train_step = train_step

    def set_custom_val_step(self, val_step):
        self.__validation_step = val_step

    def __generate_augment_dataset(self, dataset_dir:str, repeats:int):

        # Criando estrutura para o novo dataset
        classes = os.listdir(dataset_dir)
        classes.sort()

        N_CLASSES = len(classes)
        print(f"Augmentation - The current dataset have {N_CLASSES} classes")

        augmented_dataset_path = dataset_dir + f"_{N_CLASSES}_AUG"
        augmented_dataset_test_path = augmented_dataset_path + "/test/"
        augmented_dataset_train_path = augmented_dataset_path + "/train/"
        self.dataset_name = f"{self.dataset_name}_{N_CLASSES}_AUG"

        try:
            os.mkdir(augmented_dataset_path)
            os.mkdir(f"{augmented_dataset_test_path}")
            os.mkdir(f"{augmented_dataset_train_path}")

            for c in classes:
                os.mkdir(f"{augmented_dataset_train_path}/{c}/")

            # Copiando as imagens originais para o novo diretório
            for c in classes:
                imgs = os.listdir(f"{dataset_dir}/{c}")

                test_imgs = round(len(imgs) * 0.15)
                print(f"Augmentation - {test_imgs} images will be selected to from the class {c} to test!")
                selected_test_imgs_counter = 0

                # Copiando todas as imagens para o caminho de teste
                for img in imgs:
                    # Selecionando imagens aleatóriamente para colocar na pasta de teste
                    if bool(random.getrandbits(1)) and selected_test_imgs_counter < test_imgs:
                        shutil.copyfile(f"{dataset_dir}/{c}/{img}", f"{augmented_dataset_test_path}/{classes.index(c)}_{selected_test_imgs_counter}.png")
                        selected_test_imgs_counter += 1

                    shutil.copyfile(f"{dataset_dir}/{c}/{img}", f"{augmented_dataset_train_path}/{c}/{img}")
            
            j = 0
            
            for _ in range(repeats):
                # Criando as novas imagens
                preprocess = transforms.Compose([
                    transforms.ColorJitter(hue=0.05, saturation=0.05),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(20),
                ])

                dataset = datasets.ImageFolder(augmented_dataset_train_path, preprocess)
                
                for d_img, label in dataset:
                    cv2.imwrite(f"{augmented_dataset_train_path}/{classes[label]}/{j}.png", np.asarray(d_img))
                    j += 1

            print("Augmentation - The Dataset was augmented succesfully")

        except OSError as error:
            print("Augmentation - The Dataset is already augmented")

        return augmented_dataset_path
    
    def __get_dataset(self, dataset_dir:str):

        # Definindo as transformações que precisam ser feitas no conjunto de imagens
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Carregando o dataset a partir da pasta
        dataset = datasets.ImageFolder(dataset_dir + "/train/", preprocess)

        return dataset

    def __get_loaders(self, dataset):

        train, val = random_split(dataset, [0.8, 0.2])

        train_loader = torch.utils.data.DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

        return train_loader, val_loader

    def __get_optimizer(self):
        optimizer = optim.Adamax(self.model.parameters(), lr=self.lr)
        return optimizer

    def __get_criterion(self):
        return nn.CrossEntropyLoss().to(self.device)

    def procedure(self, output_folder_name:str):
    
        dataset_dir = f"../datasets/{self.dataset_name}"
        test_dataset_dir = f"../datasets/{self.dataset_name}/test/"
        model = self.model
        train_loader, val_loader = self.loaders

        final_json = {}

        # Pytorch-ignite bit
        val_metrics = {
            "accuracy": Accuracy(),
            "precision": Precision(average='macro'),
            "recall": Recall(average='macro'),
            "f1": (Precision(average=False) * Recall(average=False) * 2 / (Precision(average=False) + Recall(average=False))).mean(),
            "loss": Loss(self.criterion)
        }

        if self.__train_step is not None:
            print("Training - Using custom train step")
            trainer = Engine(self.__train_step)
        else:
            trainer = create_supervised_trainer(model, self.optimizer, self.criterion, self.device)

        if self.__validation_step is not None:
            print("Training - Using custom val step")
            val_evaluator = Engine(self.__validation_step)
        else:
            val_evaluator = create_supervised_evaluator(model, val_metrics, self.device)

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
            f"../output/{output_folder_name}/{self.dataset_name}/",
            require_empty=False,
            n_saved=1,
            filename_prefix=f"train",
            score_function=score_function,
            score_name="f1",
            global_step_transform=global_step_from_engine(trainer),
        )
        
        val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

        print(f"\nTraining the {model.__class__.__name__} model")

        trainer.run(train_loader, max_epochs=self.epochs)

        print(f"\nTrain finished for model {model.__class__.__name__}")

        with open(f"../output/{output_folder_name}/{self.dataset_name}/training_results.json", "w") as f:
            json.dump(final_json, f)

        model.load_state_dict(torch.load(model_checkpoint.last_checkpoint))

        get_cam_metrics(model, output_folder_name, self.dataset_name, test_dataset_dir)


if __name__ == "__main__":

    baseline = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    model1 = DenseNet201ABENN(baseline, N_CLASSES)

    cpf1 = ComparatorFramewok(model=model1, dataset_name="UCSB")

    def train_step(engine, batch):
            cpf1.model.train()
            cpf1.optimizer.zero_grad()
            x, y = batch[0].to(cpf1.device), batch[1].to(cpf1.device)
            y_pred = cpf1.model(x)
            loss = cpf1.criterion(y_pred, y)

            att = cpf1.model.att.detach()
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
            cpf1.optimizer.step()

            return loss.item()
        
    def validation_step(engine, batch):
        cpf1.model.eval()
        with torch.no_grad():
            x, y = batch[0].to(cpf1.device), batch[1].to(cpf1.device)
            y_pred = cpf1.model(x)
            return y_pred, y


    cpf1.set_custom_train_step(train_step)
    cpf1.set_custom_val_step(validation_step)

    cpf1.procedure("ABN")


    model2 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    model2.classifier = nn.Linear(in_features=1920, out_features=N_CLASSES, bias=True)
    model2 = DenseNetGradCam(model2)

    cpf2 = ComparatorFramewok(model=model2, dataset_name="UCSB")
    cpf2.procedure("DENSENET")
