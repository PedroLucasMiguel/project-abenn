import os
import torch
import argparse
import numpy as np
import torch.nn as nn

from src.trainables.models.resnet_abn import resnet50
from src.trainables.models.resnet_abn_cf import resnet50_cf
from src.trainables.models.densenet import DenseNet201ABENN
from src.trainables.models.densenet_abn_vit_cf_gap import DenseNet201ABNVITGAP
from src.trainables.models.resnet_abn_cf_gap import resnet50_cf_gap
from src.trainables.models.densenet_abn_cf_gap import DenseNet201ABNGAP

from src.trainables.trainer import TrainerFramework

EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001
OPTIMIZER = 'Adam'

# Attention branch
CF = 10


def train_resnet50_abn_cf(dn: str, use_gpu_n: int = 0) -> None:
    n_classes = len(os.listdir(f"../datasets/{dn}"))
    model = resnet50_cf(True, num_classes=n_classes)
    model.fc = nn.Linear(512 * model.block.expansion, n_classes)

    cpf = TrainerFramework(epochs=EPOCHS,
                           batch_size=BATCH_SIZE,
                           optimizer=OPTIMIZER,
                           lr=LEARNING_RATE,
                           model=model,
                           use_gpu_n=use_gpu_n,
                           dataset_name=dn,
                           use_augmentation=False)

    def train_step(engine, batch):
        cpf.model.train()
        cpf.optimizer.zero_grad()
        x, y = batch[0].to(cpf.device), batch[1].to(cpf.device)
        y_pred = cpf.model(x)
        loss = cpf.criterion(y_pred, y)
        att = cpf.model.att.detach()
        att = np.asarray(att)

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


def train_resnet50_abn_cf_gap(dn: str, use_gpu_n: int = 0) -> None:
    n_classes = len(os.listdir(f"../datasets/{dn}"))
    model = resnet50_cf_gap(True, num_classes=n_classes)
    model.fc = nn.Linear(512 * model.block.expansion, n_classes)

    cpf = TrainerFramework(epochs=EPOCHS,
                           batch_size=BATCH_SIZE,
                           optimizer=OPTIMIZER,
                           lr=LEARNING_RATE,
                           model=model,
                           use_gpu_n=use_gpu_n,
                           dataset_name=dn,
                           use_augmentation=False)

    def train_step(engine, batch):
        cpf.model.train()
        cpf.optimizer.zero_grad()
        x, y = batch[0].to(cpf.device), batch[1].to(cpf.device)
        y_pred = cpf.model(x)
        loss = cpf.criterion(y_pred, y)
        att = cpf.model.att.detach()
        att = np.asarray(att)

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

    cpf.procedure("RESNET_ABN_CF_GAP")


def train_densenet201_abn_cf(dn: str, use_gpu_n: int = 0) -> None:
    n_classes = len(os.listdir(f"../datasets/{dn}"))
    baseline = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    model = DenseNet201ABENN(baseline, n_classes, False)

    cpf = TrainerFramework(epochs=EPOCHS,
                           batch_size=BATCH_SIZE,
                           optimizer=OPTIMIZER,
                           lr=LEARNING_RATE,
                           model=model,
                           use_gpu_n=use_gpu_n,
                           dataset_name=dn,
                           use_augmentation=False)

    def train_step(engine, batch):
        cpf.model.train()
        cpf.optimizer.zero_grad()
        x, y = batch[0].to(cpf.device), batch[1].to(cpf.device)
        y_pred = cpf.model(x)
        loss = cpf.criterion(y_pred, y)

        att = cpf.model.att.detach()
        att = np.asarray(att)
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

    cpf.procedure("DENSENET_ABN_CF")


def train_densenet201_abn_vit_cf_gap(dn: str, use_gpu_n: int = 0) -> None:
    n_classes = len(os.listdir(f"../datasets/{dn}"))
    baseline = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    model = DenseNet201ABNVITGAP(baseline, n_classes, False)

    cpf = TrainerFramework(epochs=EPOCHS,
                           batch_size=BATCH_SIZE,
                           optimizer=OPTIMIZER,
                           lr=LEARNING_RATE,
                           model=model,
                           use_gpu_n=use_gpu_n,
                           dataset_name=dn,
                           use_augmentation=False)

    def train_step(engine, batch):
        cpf.model.train()
        cpf.optimizer.zero_grad()
        x, y = batch[0].to(cpf.device), batch[1].to(cpf.device)
        y_pred = cpf.model(x)
        loss = cpf.criterion(y_pred, y)

        att = cpf.model.att.detach()
        att = np.asarray(att)
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

    cpf.procedure("DENSENET_ABN_CF_GAP")


def train_densenet201_abn_cf_gap(dn: str, use_gpu_n: int = 0) -> None:
    n_classes = len(os.listdir(f"../datasets/{dn}"))
    baseline = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)
    model = DenseNet201ABNGAP(baseline, n_classes, False)

    cpf = TrainerFramework(epochs=EPOCHS,
                           batch_size=BATCH_SIZE,
                           optimizer=OPTIMIZER,
                           lr=LEARNING_RATE,
                           model=model,
                           use_gpu_n=use_gpu_n,
                           dataset_name=dn,
                           use_augmentation=False)

    def train_step(engine, batch):
        cpf.model.train()
        cpf.optimizer.zero_grad()
        x, y = batch[0].to(cpf.device), batch[1].to(cpf.device)
        y_pred = cpf.model(x)
        loss = cpf.criterion(y_pred, y)

        att = cpf.model.att.detach()
        att = np.asarray(att)
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

    cpf.procedure("DENSENET_ABN_VIT_CF_GAP")


def train_resnet50_abn(dn: str, use_gpu_n: int = 0) -> None:
    n_classes = len(os.listdir(f"../datasets/{dn}"))
    model = resnet50(True, num_classes=n_classes)
    model.fc = nn.Linear(512 * model.block.expansion, n_classes)

    cpf = TrainerFramework(epochs=EPOCHS,
                           batch_size=BATCH_SIZE,
                           optimizer=OPTIMIZER,
                           lr=LEARNING_RATE,
                           model=model,
                           use_gpu_n=use_gpu_n,
                           dataset_name=dn,
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

    cpf.set_custom_train_step(train_step)
    cpf.set_custom_val_step(validation_step)
    cpf.procedure("RESNET_ABN")


if __name__ == "__main__":

    valid_models = ['RESNET50_ABN_CF',
                    'RESNET50_ABN_CF_GAP',
                    'DENSENET201_ABN_CF',
                    'DENSENET201_ABN_CF_GAP',
                    'DENSENET201_ABN_VIT_CF_GAP',
                    'RESNET50_ABN']

    # Datasets
    dts1 = ["CR", "LA", "LG", "NHL", "UCSB"]

    parser = argparse.ArgumentParser(prog='Comparator Framework',
                                     description='This Program enable to compare the explanation os two models.')

    parser.add_argument('-m', '--models', type=str, required=True)
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-co', '--clearoutput', type=bool, required=False)

    args = parser.parse_args()

    if args.gpu < 0 or args.gpu > 1:
        print("Invalid GPU ID")

    if args.model not in valid_models:
        print("Invalid Model!")
        print(f"Valid Options = {valid_models}")

    if args.clearoutput:
        print("Clearing previous outputs...")
        os.system('rm -rf ../output/*')

    for dn in dts1:
        match args.model:
            case 'RESNET50_ABN_CF':
                train_resnet50_abn_cf(dn, args.gpu)
            case 'DENSENET201_ABN_CF':
                train_densenet201_abn_cf(dn, args.gpu)
            case 'DENSENET201_ABN_CF_GAP':
                train_densenet201_abn_cf_gap(dn, args.gpu)
            case 'DENSENET201_ABN_VIT_CF_GAP':
                train_densenet201_abn_vit_cf_gap(dn, args.gpu)
            case 'RESNET50_ABN':
                train_resnet50_abn(dn, args.gpu)
            case 'RESNET50_ABN_CF_GAP':
                train_resnet50_abn_cf_gap(dn, args.gpu)

        torch.cuda.empty_cache()
