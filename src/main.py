import os
import argparse
from trainables.resnet import *
from trainables.densenet import *

if __name__ == '__main__':

    valid_models = ['RESNET50_ABN_CF',
                    'RESNET50_ABN_CF_GAP',
                    'DENSENET201_ABN_CF',
                    'DENSENET201_ABN_CF_GAP',
                    'DENSENET201_ABN_VIT_CF_GAP',
                    'RESNET50_ABN']

    # Datasets
    #dts1 = ['CR', 'LA', 'LG', 'NHL', 'UCSB']
    dts1 = ['LA']

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
                trainable = ResNet50ABNCFGAP(dataset_name=dn)
                trainable.procedure('RESNET50_ABN_CF_GAP')
            case 'RESNET50_ABN':
                trainable = ResNet50ABN(dataset_name=dn)
                trainable.procedure('RESNET50_ABN')
            case 'RESNET50_ABN_CF':
                trainable = ResNet50ABNCF(dataset_name=dn)
                trainable.procedure('RESNET50_ABN_CF')
            case 'DENSENET201_ABN_CF':
                trainable = DenseNet201ABNCF(dataset_name=dn)
                trainable.procedure('DENSENET201_ABN_CF')
            case 'DENSENET201_ABN_CF_GAP':
                trainable = DenseNet201ABNCFGAP(dataset_name=dn)
                trainable.procedure('DENSENET201_ABN_CF_GAP')
            case 'DENSENET201_ABN_VIT_CF_GAP':
                trainable = TrainableDenseNet201ABNVITCFGAP(dataset_name=dn)
                trainable.procedure('DENSENET201_ABN_VIT_CF_GAP')