import os
import torch
import argparse
from trainables.resnet import *
from trainables.densenet import *
from trainables.efficientnet import *
from trainables.coatnet import *
from trainables.convnext import *
from trainables.resnext import *
from trainables.uniformer import *
from trainables.global_config import REPEATED_HOLDOUT_REPEATS

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
                    'EFFICIENTNET_ABN_CF_GAP',
                    'COATNETB0']

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
        torch.cuda.empty_cache()
        match args.model:
            case 'RESNET50_ABN_CF_GAP':
                for i in range(REPEATED_HOLDOUT_REPEATS):
                    trainable = TrainableResNet50ABNCFGAP(dataset_name=dn)
                    trainable.procedure_repeated_holdout('RESNET50_ABN_CF_GAP', i)
            case 'RESNET50_ABN':
                trainable = TrainableResNet50ABN(dataset_name=dn)
                trainable.new_proceadure('RESNET50_ABN')
            case 'DENSENET201_ABN':
                trainable = TrainableDenseNet201ABN(dataset_name=dn)
                trainable.new_proceadure('DENSENET201_ABN')
            case 'DENSENET201_ABN_CF_GAP':
                for i in range(REPEATED_HOLDOUT_REPEATS):
                    trainable = TrainableDenseNet201ABNCFGAP(dataset_name=dn)
                    trainable.procedure_repeated_holdout('DENSENET201_ABN_CF_GAP', i)
            case 'DENSENET201_ABN_VIT_CF_GAP':
                trainable = TrainableEfficientNetABNCFGAP(dataset_name=dn)
                trainable.new_proceadure('DENSENET201_ABN_VIT_CF_GAP')
            case 'RESNET50':
                for i in range(REPEATED_HOLDOUT_REPEATS):
                    trainable = ResNet50Baseline(dataset_name=dn)
                    trainable.procedure_repeated_holdout('RESNET50', i)
            case 'DENSENET201':
                for i in range(REPEATED_HOLDOUT_REPEATS):
                    trainable = TrainableDenseNet201Baseline(dataset_name=dn)
                    trainable.procedure_repeated_holdout('DENSENET201', i)
            case 'EFFICIENTNET':
                for i in range(REPEATED_HOLDOUT_REPEATS):
                    trainable = TrainableEfficientNetBaseline(dataset_name=dn)
                    trainable.procedure_repeated_holdout('EFFICIENTNET', i)
            case 'EFFICIENTNET_ABN_CF_GAP':
                for i in range(REPEATED_HOLDOUT_REPEATS):
                    trainable = TrainableEfficientNetABNCFGAP(dataset_name=dn)
                    trainable.procedure_repeated_holdout('EFFICIENTNET_ABN_CF_GAP', i)
            case 'COATNETB0':
                for i in range(REPEATED_HOLDOUT_REPEATS):
                    trainable = TrainableCoatNetBaseline(dataset_name=dn)
                    trainable.procedure_repeated_holdout('COATNETB0', i)
            case 'COATNET_ABN_CF_GAP':
                for i in range(REPEATED_HOLDOUT_REPEATS):
                    trainable = TrainableCoatNetABNCFGAP(dataset_name=dn)
                    trainable.procedure_repeated_holdout('COATNET_ABN_CF_GAP', i)
            case 'CONVNEXT_ABN_CF_GAP':
                for i in range(REPEATED_HOLDOUT_REPEATS):
                    trainable = TrainableConvNextABNCFGAP(dataset_name=dn)
                    trainable.procedure_repeated_holdout('CONVNEXT_ABN_CF_GAP', i)
            case 'CONVNEXT_SMALL':
                for i in range(REPEATED_HOLDOUT_REPEATS):
                    trainable = TrainableConvNextSmall(dataset_name=dn)
                    trainable.procedure_repeated_holdout('CONVNEXT_SMALL', i)
            case 'RESNEXT50':
                for i in range(REPEATED_HOLDOUT_REPEATS):
                    trainable = TrainableResNextBaseline(dataset_name=dn)
                    trainable.procedure_repeated_holdout('RESNEXT50', i)
            case 'RESNEXT_ABN_CF_GAP':
                for i in range(REPEATED_HOLDOUT_REPEATS):
                    trainable = TrainableResNet50ABNCFGAP(dataset_name=dn)
                    trainable.procedure_repeated_holdout('RESNEXT50_ABN_CF_GAP', i)
            case 'UNIFORMER_BASELINE':
                for i in range(REPEATED_HOLDOUT_REPEATS):
                    trainable = TrainableUniformerBaseline(dataset_name=dn)
                    trainable.procedure_repeated_holdout('UNIFORMER_BASELINE', i)
            case 'UNIFORMER_ABN_CF_GAP':
                for i in range(REPEATED_HOLDOUT_REPEATS):
                    trainable = TrainableUniformerABNCFGAP(dataset_name=dn)
                    trainable.procedure_repeated_holdout('UNIFORMER_ABN_CF_GAP', i)