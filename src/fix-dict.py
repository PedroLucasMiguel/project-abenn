import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import warnings
import cv2
from PIL import Image
from torchvision import transforms
from model.densenet import DenseNet201ABENN
warnings.filterwarnings("ignore", category=UserWarning) 

f = torch.load("checkpoints/e_7_covid_savestate.pth")

new_dict = {}

for key, value in f.items():
    if 'module.' in key:
        key = key.replace("module.", "")
    new_dict[key] = value

torch.save(new_dict, "best_covid_full.pth")