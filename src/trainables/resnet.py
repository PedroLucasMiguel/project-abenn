import os
import numpy as np
from torch import nn, no_grad
from .trainer.trainer import TrainerFramework
from .models.resnet_abn_cf_gap import resnet50_cf_gap
from .models.resnet_abn import resnet50
from .global_config import *


class ResNet50ABN(TrainerFramework):
    def __init__(self, dataset_name: str) -> None:
        self.n_classes = len(os.listdir(os.path.join('..', 'datasets', dataset_name)))
        self.trainable_model = resnet50(True, num_classes=self.n_classes)
        self.trainable_model.fc = nn.Linear(512 * self.trainable_model.block.expansion, self.n_classes)

        super().__init__(epochs=EPOCHS,
                         batch_size=BATCH_SIZE,
                         optimizer=OPTIMIZER,
                         lr=LEARNING_RATE,
                         model=self.trainable_model,
                         dataset_name=dataset_name,
                         use_augmentation=False)

    def train_step(engine, batch):
        cpf.model.train()
        cpf.optimizer.zero_grad()
        x, y = batch[0].to(cpf.device), batch[1].to(cpf.device)
        att_outputs, outputs, _ = cpf.model(x)

        att_loss = cpf.criterion(att_outputs, y)
        per_loss = cpf.criterion(outputs, y)
        loss = att_loss + per_loss

        loss.backward()
        cpf.optimizer.step()

        return loss.item()

    def validation_step(engine, batch):
        cpf.model.eval()
        with torch.no_grad():
            x, y = batch[0].to(cpf.device), batch[1].to(cpf.device)
            _, y_pred, _ = cpf.model(x)
            return y_pred, y


class ResNet50ABNCFGAP(TrainerFramework):
    def __init__(self, dataset_name: str) -> None:
        self.n_classes = len(os.listdir(os.path.join('..', 'datasets', dataset_name)))
        self.trainable_model = resnet50_cf_gap(True, num_classes=self.n_classes)
        self.trainable_model.fc = nn.Linear(512 * self.trainable_model.block.expansion, self.n_classes)

        super().__init__(epochs=EPOCHS,
                         batch_size=BATCH_SIZE,
                         optimizer=OPTIMIZER,
                         lr=LEARNING_RATE,
                         model=self.trainable_model,
                         dataset_name=dataset_name,
                         use_augmentation=False)

    def train_step(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        att = np.asarray(self.model.att.detach())

        cam_normalized = np.zeros((att.shape[0], att.shape[2], att.shape[3]))

        for i in range(att.shape[0]):
            s = np.sum(att[i, 0, :, :])
            cam_normalized[i, :, :] = np.divide(att[i, 0, :, :], s)

        # Realizando a média dos batches
        #m = np.mean(cam_normalized, axis=0)
        #ce = CF*np.sum(m*np.log(m))

        cam_normalized_log = np.log(cam_normalized)
        cam_sum = np.sum(np.multiply(cam_normalized, cam_normalized_log))
        ce = CF * cam_sum
        loss = loss - ce.item()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def val_step(self, engine, batch):
        self.model.eval()
        with no_grad():
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            y_pred = self.model(x)
            return y_pred, y
