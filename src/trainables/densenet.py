import os
import numpy as np
from torch import hub, no_grad
from .trainer.trainer import TrainerFramework
from .models.densenet_abn import DenseNet201ABENN
from .models.densenet_abn_cf_gap import DenseNet201ABNGAP
from .models.densenet_abn_vit_cf_gap import DenseNet201ABNVITGAP
from .global_config import *
from .models.densenet_baseline import DenseNetGradCam


class TrainableDenseNet201ABN(TrainerFramework):
    def __init__(self, dataset_name: str) -> None:
        self.n_classes = len(os.listdir(f"../datasets/{dataset_name}"))
        self.baseline = hub.load(
            'pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
        self.trainable_model = DenseNet201ABENN(self.baseline, self.n_classes, False)

        super().__init__(epochs=EPOCHS,
                         batch_size=BATCH_SIZE,
                         optimizer=OPTIMIZER,
                         lr=LEARNING_RATE,
                         model=self.trainable_model,
                         dataset_name=dataset_name,
                         use_augmentation=False)

    def train_step(self, engine, batch):
        self.trainable_model.train()
        self.optimizer.zero_grad()
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        att_outputs, outputs, _ = self.trainable_model(x)

        att_loss = self.criterion(att_outputs, y)
        per_loss = self.criterion(outputs, y)
        loss = att_loss + per_loss

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def val_step(self, engine, batch):
        self.trainable_model.eval()
        with no_grad():
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            _, y_pred, _ = self.trainable_model(x)
            return y_pred, y


class TrainableDenseNet201ABNCFGAP(TrainerFramework):
  def __init__(self, dataset_name: str) -> None:
    self.n_classes = len(os.listdir(f"../datasets/{dataset_name}"))
    self.baseline = hub.load('pytorch/vision:v0.10.0',
                             'densenet201', pretrained=True)
    self.trainable_model = DenseNet201ABNGAP(self.baseline, self.n_classes, False)

    super().__init__(epochs=EPOCHS,
                         batch_size=BATCH_SIZE,
                         optimizer=OPTIMIZER,
                         lr=LEARNING_RATE,
                         model=self.trainable_model,
                         dataset_name=dataset_name,
                         use_augmentation=False)

  def train_step(self, engine, batch):
    self.trainable_model.train()
    self.optimizer.zero_grad()
    x, y = batch[0].to(self.device), batch[1].to(self.device)
    y_pred = self.trainable_model(x)
    loss = self.criterion(y_pred, y)

    att = self.trainable_model.att.detach().cpu()
    att = np.asarray(att)
    cam_normalized = np.zeros((att.shape[0], att.shape[2], att.shape[3]))

    for i in range(att.shape[0]):
      s = np.sum(att[i, 0, :, :])
      cam_normalized[i, :, :] = np.divide(att[i, 0, :, :], s)

    # Realizando a média dos batches
    # m = np.mean(cam_normalized, axis=0)
    # ce = CF*np.sum(m*np.log(m))

    cam_normalized_log = np.log(cam_normalized)
    cam_sum = np.sum(np.multiply(cam_normalized, cam_normalized_log))
    ce = CF * cam_sum

    loss = loss - ce.item()

    loss.backward()
    self.optimizer.step()

    return loss.item()

  def val_step(self, engine, batch):
    self.trainable_model.eval()
    with no_grad():
      x, y = batch[0].to(self.device), batch[1].to(self.device)
      y_pred = self.trainable_model(x)
      return y_pred, y

class TrainableDenseNet201ABNVITCFGAP(TrainerFramework):
  def __init__(self, dataset_name: str) -> None:
    self.n_classes = len(os.listdir(f"../datasets/{dataset_name}"))
    self.baseline = hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    self.trainable_model = DenseNet201ABNVITGAP(baseline_model=self.baseline, n_classes=self.n_classes, freeze_training=False)

    super().__init__(epochs=EPOCHS,
                         batch_size=BATCH_SIZE,
                         optimizer=OPTIMIZER,
                         lr=LEARNING_RATE,
                         model=self.trainable_model,
                         dataset_name=dataset_name,
                         use_augmentation=False)

  def train_step(self, engine, batch):
    self.trainable_model.train()
    self.optimizer.zero_grad()
    x, y = batch[0].to(self.device), batch[1].to(self.device)
    y_pred = self.trainable_model(x)
    loss = self.criterion(y_pred, y)

    att = self.trainable_model.att.detach().cpu()
    att = np.asarray(att)
    cam_normalized = np.zeros((att.shape[0], att.shape[2], att.shape[3]))

    for i in range(att.shape[0]):
      s = np.sum(att[i, 0, :, :])
      cam_normalized[i, :, :] = np.divide(att[i, 0, :, :], s)

    # # Realizando a média dos batches
    # #m = np.mean(cam_normalized, axis=0)
    # #ce = CF*np.sum(m*np.log(m))

    cam_normalized_log = np.log(cam_normalized)
    cam_sum = np.sum(np.multiply(cam_normalized, cam_normalized_log))
    ce = CF * cam_sum

    loss = loss - ce.item()

    loss.backward()
    self.optimizer.step()

    return loss.item()

  def val_step(self, engine, batch):
    self.trainable_model.eval()
    with no_grad():
      x, y = batch[0].to(self.device), batch[1].to(self.device)
      y_pred = self.trainable_model(x)
      return y_pred, y

class TrainableDenseNet201Baseline(TrainerFramework):
  def __init__(self, dataset_name: str) -> None:
    self.n_classes = len(os.listdir(f"../datasets/{dataset_name}"))
    self.baseline = hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    self.trainable_model = DenseNetGradCam(self.baseline, n_classes=self.n_classes)

    super().__init__(epochs=EPOCHS,
                         batch_size=BATCH_SIZE,
                         optimizer=OPTIMIZER,
                         lr=LEARNING_RATE,
                         model=self.trainable_model,
                         dataset_name=dataset_name,
                         use_augmentation=False)

  def train_step(self, engine, batch):
    self.trainable_model.train()
    self.optimizer.zero_grad()
    x, y = batch[0].to(self.device), batch[1].to(self.device)
    y_pred = self.trainable_model(x)
    loss = self.criterion(y_pred, y)

    loss.backward()
    self.optimizer.step()

    return loss.item()

  def val_step(self, engine, batch):
    self.trainable_model.eval()
    with no_grad():
      x, y = batch[0].to(self.device), batch[1].to(self.device)
      y_pred = self.trainable_model(x)
      return y_pred, y