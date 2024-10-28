import os
import argparse
from trainables.resnet import *
from trainables.densenet import *
from trainables.efficientnet import *

if __name__ == '__main__':

    valid_models = ['RESNET50_ABN_CF',
                    'RESNET50_ABN_CF_GAP',
                    'DENSENET201_ABN',
                    'DENSENET201_ABN_CF_GAP',
                    'DENSENET201_ABN_VIT_CF_GAP',
                    'RESNET50_ABN'
                    'RESNET50',
                    'DENSENET201',
                    'EFFICIENTNET',
                    'EFFICIENTNET_ABN_CF_GAP']

    # Datasets
    dts1 = ['CR', 'LA', 'LG', 'NHL', 'UCSB']
    #dts1 = ['UCSB']

    parser = argparse.ArgumentParser(prog='Comparator Framework',
                                     description='This Program enable to compare the explanation os two models.')

    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-co', '--clearoutput', type=bool, required=False)

    args = parser.parse_args()

    if args.model not in valid_models:
        print('Invalid Model!')
        print(f'Valid Options = {valid_models}')

    if args.clearoutput:
        print('Clearing previous outputs...')
        os.system(f'rm -rf {os.path.join("..", "output", "*")}')

    for dn in dts1:
        match args.model:
            case 'RESNET50_ABN_CF_GAP':
                trainable = TrainableResNet50ABNCFGAP(dataset_name=dn)
                trainable.procedure('RESNET50_ABN_CF_GAP')
            case 'RESNET50_ABN':
                trainable = TrainableResNet50ABN(dataset_name=dn)
                trainable.procedure('RESNET50_ABN')
            case 'DENSENET201_ABN':
                trainable = TrainableDenseNet201ABN(dataset_name=dn)
                trainable.procedure('DENSENET201_ABN')
            case 'DENSENET201_ABN_CF_GAP':
                trainable = TrainableDenseNet201ABNCFGAP(dataset_name=dn)
                trainable.procedure('DENSENET201_ABN_CF_GAP')
            case 'DENSENET201_ABN_VIT_CF_GAP':
                trainable = TrainableEfficientNetABNCFGAP(dataset_name=dn)
                trainable.procedure('DENSENET201_ABN_VIT_CF_GAP')
            case 'RESNET50':
                trainable = ResNet50Baseline(dataset_name=dn)
                trainable.procedure('RESNET50')
            case 'DENSENET201':
                trainable = TrainableDenseNet201Baseline(dataset_name=dn)
                trainable.procedure('DENSENET201')
            case 'EFFICIENTNET':
                trainable = TrainableEfficientNetBaseline(dataset_name=dn)
                trainable.procedure('EFFICIENTNET')
            case 'EFFICIENTNET_ABN_CF_GAP':
                trainable = TrainableEfficientNetABNCFGAP(dataset_name=dn)
                trainable.procedure('EFFICIENTNET_ABN_CF_GAP')