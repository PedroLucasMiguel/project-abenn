import os
import cv2
import json
import random
import shutil
import numpy as np

import torch
from torch import nn
from torch import load
from torch import cuda
from torch import backends
from typing import Any
from torch import optim
from abc import ABC, abstractmethod
from torch.utils import data
from torch.optim import Optimizer
from torchvision import datasets
from torchvision import transforms
from .cam_metrics import get_cam_metrics
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Accuracy, Precision, Recall, Loss
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator


class TrainerFramework(ABC):
    def __init__(self, epochs: int = 10, batch_size: int = 16, optimizer: str = 'Adamax', lr: float = 0.0001,
                 weight_decay: float = 0.1, momentum: float = 0.9, model: nn.Module = None, dataset_name: str = None,
                 use_augmentation: bool = True, get_test_metrics: bool = False, augmentation_repeats: int = 2,
                 use_repeated_holdout: bool = True, repeatet_holdout_iterations: int = 5) -> None:

        # Hyper parameters
        self.lr = lr
        self.epochs = epochs
        self.momentum = momentum
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        # Model configuration and optimizer configuration
        if cuda.is_available():
            self.device = 'cuda'
        elif backends.mps.is_available():
            if backends.mps.is_built():
                self.device = 'mps'
        else:
            self.device = 'cpu'

        # self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.model = model.to(device=self.device)
        self.optimizer = self.__get_optimizer(optm_type=optimizer)
        # if self.device == 'cpu':
        #     self.model, self.optimizer = ipex.optimize(model=model, optimizer=self.optimizer)

        # Training configuration
        self.criterion = self.__get_criterion()

        # Dataset definition
        self.dataset_name = dataset_name
        self.use_augmentation = use_augmentation
        self.get_test_metrics = get_test_metrics
        self.use_repeated_holdout = use_repeated_holdout
        self.repeated_holdout_iterations = repeatet_holdout_iterations

        # Verifying augmentation
        if self.use_augmentation:
            self.loaders = self.__get_loaders(
                self.__get_dataset(
                    self.__generate_augment_dataset(os.path.join('..', 'datasets', self.dataset_name), augmentation_repeats)))
        elif self.get_test_metrics:
            self.loader = self.__get_test_loader(self.__get_test_dataset(
                os.path.join('..', 'datasets', 'fdp', self.dataset_name)))
        elif self.use_repeated_holdout:
            self.loaders = self.__get_loaders_repeated_holdout(
                self.__get_datasets_repeated_holdout(self.__convert_datasets_repeated_holdout(os.path.join('..', 'datasets', self.dataset_name))))
        else:
            self.loaders = self.__get_loaders(
                self.__get_dataset(self.__convert_dataset(os.path.join('..', 'datasets', self.dataset_name))))

    def __convert_datasets_repeated_holdout(self, dataset_dir: str):

        # Getting the class names
        classes = os.listdir(dataset_dir)
        classes.sort()

        n_classes = len(classes)
        print(
            f'Trainer Framework - The current dataset have {n_classes} classes')

        dataset_paths = []
        self.dataset_name = f"{self.dataset_name}_{n_classes}_CV"

        for i in range(self.repeated_holdout_iterations):
            # Defining the dataset structure
            dataset_path = dataset_dir + \
                f'_{n_classes}_CV_{i}'
            dataset_paths.append(dataset_path)
            dataset_train_path = os.path.join(dataset_path, 'train')
            dataset_test_path = os.path.join(dataset_path, 'test')
            dataset_val_path = os.path.join(dataset_path, 'val')

            try:
                # Creating folders
                os.mkdir(dataset_path)
                os.mkdir(dataset_train_path)
                os.mkdir(dataset_test_path)
                os.mkdir(dataset_val_path)

                # Calculating the number os samples in each dataset
                n_samples = 0
                for c in classes:
                    os.mkdir(os.path.join(dataset_train_path, c))
                    os.mkdir(os.path.join(dataset_val_path, c))
                    n_samples += len(os.listdir(os.path.join(dataset_dir, c)))

                print(
                    f'Trainer Framework - N° of images in the dataset: {n_samples}')

                # Calculating the number of images that must be present in the validation step
                # For now, we just assume a 70/15/15 split
                val_split = test_split = round(n_samples * 0.30) / 2

                val_samples_counter = 0
                test_samples_counter = 0

                '''
                    The next loop do the following:
                    1° Randomly select one of the classes in the dataset
                    2° Randomly selects if the image will be assign to the validation or test sets
                    3° If we filled the val/test sets, or the image "failed" in both tests, assign the image to the train
                    set
                '''
                for _ in range(n_samples):
                    c = random.choice(classes)
                    samples = os.listdir(f"{dataset_dir}/{c}")
                    img = np.random.choice(samples)

                    # Testing for validation
                    if bool(random.getrandbits(1)) and val_samples_counter < val_split:
                        shutil.copyfile(os.path.join(dataset_dir, c, img),
                                        os.path.join(dataset_val_path, c, img))
                        val_samples_counter += 1
                        continue

                    # Testing for tes
                    elif bool(random.getrandbits(1)) and test_samples_counter < test_split:
                        shutil.copyfile(os.path.join(dataset_dir, c, img),
                                        os.path.join(dataset_test_path, f'{classes.index(c)}_{test_samples_counter}.png'))
                        test_samples_counter += 1
                        continue

                    # Assign to the training set
                    shutil.copyfile(os.path.join(dataset_dir, c, img),
                                    os.path.join(dataset_train_path, c, img))

            # If we fail to create the folders, we just assume that the dataset is already converted
            except OSError as _:
                print('Trainer Framework - The Dataset is already augmented')

        return dataset_paths

    def __convert_dataset(self, dataset_dir: str, suffix: str = None):

        # Getting the class names
        classes = os.listdir(dataset_dir)
        classes.sort()

        n_classes = len(classes)
        print(
            f'Trainer Framework - The current dataset have {n_classes} classes')

        # Defining the dataset structure
        dataset_path = dataset_dir + \
            f'_{n_classes}_CV{"_" + suffix if suffix else ""}'
        dataset_train_path = os.path.join(dataset_path, 'train')
        dataset_test_path = os.path.join(dataset_path, 'test')
        dataset_val_path = os.path.join(dataset_path, 'val')
        self.dataset_name = f"{self.dataset_name}_{n_classes}_CV"

        try:
            # Creating folders
            os.mkdir(dataset_path)
            os.mkdir(dataset_train_path)
            os.mkdir(dataset_test_path)
            os.mkdir(dataset_val_path)

            # Calculating the number os samples in each dataset
            n_samples = 0
            for c in classes:
                os.mkdir(os.path.join(dataset_train_path, c))
                os.mkdir(os.path.join(dataset_val_path, c))
                n_samples += len(os.listdir(os.path.join(dataset_dir, c)))

            print(
                f'Trainer Framework - N° of images in the dataset: {n_samples}')

            # Calculating the number of images that must be present in the validation step
            # For now, we just assume a 70/15/15 split
            val_split = test_split = round(n_samples * 0.30) / 2

            val_samples_counter = 0
            test_samples_counter = 0

            '''
                The next loop do the following:
                1° Randomly select one of the classes in the dataset
                2° Randomly selects if the image will be assign to the validation or test sets
                3° If we filled the val/test sets, or the image "failed" in both tests, assign the image to the train
                   set
            '''
            for _ in range(n_samples):
                c = random.choice(classes)
                samples = os.listdir(f"{dataset_dir}/{c}")
                img = np.random.choice(samples)

                # Testing for validation
                if bool(random.getrandbits(1)) and val_samples_counter < val_split:
                    shutil.copyfile(os.path.join(dataset_dir, c, img),
                                    os.path.join(dataset_val_path, c, img))
                    val_samples_counter += 1
                    continue

                # Testing for tes
                elif bool(random.getrandbits(1)) and test_samples_counter < test_split:
                    shutil.copyfile(os.path.join(dataset_dir, c, img),
                                    os.path.join(dataset_test_path, f'{classes.index(c)}_{test_samples_counter}.png'))
                    test_samples_counter += 1
                    continue

                # Assign to the training set
                shutil.copyfile(os.path.join(dataset_dir, c, img),
                                os.path.join(dataset_train_path, c, img))

        # If we fail to create the folders, we just assume that the dataset is already converted
        except OSError as _:
            print('Trainer Framework - The Dataset is already augmented')

        return dataset_path

    # This method to basically the same thing as __convert_dataset(), but with augmentation in mind
    def __generate_augment_dataset(self, dataset_dir: str, repeats: int) -> str:

        try:
            aug_dataset_path = self.__convert_dataset(
                dataset_dir=dataset_dir, suffix='augmented')
            classes = os.listdir(aug_dataset_path)
            aug_dataset_train_path = os.path.join(aug_dataset_path, 'train')

            c = 0
            for _ in range(repeats):
                # Defining the transformations for the augmentation
                preprocess = transforms.Compose([
                    transforms.ColorJitter(hue=0.05, saturation=0.05),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(20),
                ])

                dataset = datasets.ImageFolder(
                    aug_dataset_train_path, preprocess)

                for d_img, label in dataset:
                    cv2.imwrite(
                        f"{aug_dataset_train_path}/{classes[label]}/{c}.png", np.asarray(d_img))
                    c += 1

            print('Trainer Framework - The Dataset was augmented successfully')
            return aug_dataset_path

        except OSError as _:
            print('Trainer Framework  - The Dataset is already augmented')

    @staticmethod
    def __get_dataset(dataset_dir: str):
        # Defining the transformations for the dataset
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        # Loading the dataset
        return (datasets.ImageFolder(os.path.join(dataset_dir, 'train'), preprocess),
                datasets.ImageFolder(os.path.join(dataset_dir, 'val'), preprocess))

    @staticmethod
    def __get_datasets_repeated_holdout(dataset_dirs: list[str]):
        # Defining the transformations for the dataset
        datasets_to_return = []
        for dataset_dir in dataset_dirs:
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225]),
            ])

            # Loading the dataset
            datasets_to_return.append((datasets.ImageFolder(os.path.join(dataset_dir, 'train'), preprocess),
                                       datasets.ImageFolder(os.path.join(dataset_dir, 'val'), preprocess)))
        return datasets_to_return

    @staticmethod
    def __get_test_dataset(dataset_dir: str):
        # Defining the transformations for the dataset
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        # Loading the dataset
        return datasets.ImageFolder(dataset_dir, preprocess)

    def __get_loaders(self, dataset):
        # Creating trainer loader
        train_loader = data.DataLoader(
            dataset[0],
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

        # Creating val loader
        val_loader = data.DataLoader(
            dataset[1],
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )

        return train_loader, val_loader
    
    def __get_loaders_repeated_holdout(self, datasets):
        # Creating trainer loader
        loaders = []
        for dataset in datasets:
            train_loader = data.DataLoader(
                dataset[0],
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
            )

            # Creating val loader
            val_loader = data.DataLoader(
                dataset[1],
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
            )

            loaders.append((train_loader, val_loader))

        return loaders

    @staticmethod
    def __get_test_loader(dataset):
        test_loader = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            pin_memory=True,
        )
        return test_loader

    def __get_optimizer(self, optm_type: str = 'Adamax') -> Optimizer:
        match optm_type:
            case 'Adam':
                return optim.Adam(self.model.parameters(), lr=self.lr)
            case 'AdamW':
                return optim.AdamW(self.model.parameters(), lr=self.lr)
            case 'Adamax':
                return optim.Adamax(self.model.parameters(), lr=self.lr)
            case 'RMSprop':
                return optim.RMSprop(self.model.parameters(), lr=self.lr)
            case 'SGD':
                return optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                 momentum=self.momentum)
            case _:
                print(
                    'Trainer Framework: Optimizer not supported. Defaulting to Adamax...')
                return optim.Adamax(self.model.parameters(), lr=self.lr)

    def __get_criterion(self):
        # TODO - Maybe enable support for other criterion methods
        return nn.CrossEntropyLoss().to(self.device)

    @abstractmethod
    def train_step(self, engine: Engine, batch: Any):
        pass

    @abstractmethod
    def val_step(self, engine: Engine, batch: Any):
        pass

    def procedure_repeated_holdout(self, output_folder_name: str, dt_index: int):
        model = self.model
        train_loader, val_loader = self.loaders[dt_index]

        final_json = {}

        # Pytorch-ignite bit
        val_metrics = {
            "accuracy": Accuracy(device=self.device),
            "precision": Precision(average='weighted', device=self.device),
            "recall": Recall(average='weighted', device=self.device),
            "f1": (Precision(average='weighted', device=self.device) * Recall(average='weighted', device=self.device) * 2 / (
                Precision(average='weighted', device=self.device) + Recall(average='weighted', device=self.device))),
            "loss": Loss(self.criterion, device=self.device)
        }

        # Here, we check if a method is actually not overwritten by the class
        trainer = Engine(self.train_step)
        validator = Engine(self.val_step)

        '''
        if len(self.__abstractmethods__) > 0:
            for f in self.__abstractmethods__:
                if f == 'train_step':
                    trainer = create_supervised_trainer(
                        model, self.optimizer, self.criterion, self.device)
                elif f == 'val_step':
                    validator = create_supervised_evaluator(
                        model, val_metrics, self.device)
        '''

        # Attaching metrics
        for name, metric in val_metrics.items():
            metric.attach(validator, name)

        # Creating the loading bards
        train_bar = ProgressBar(desc="Training...")
        val_bar = ProgressBar(desc="Evaluating...")
        train_bar.attach(trainer)
        val_bar.attach(validator)

        # Defining that after each epoch, we should run the validator once and show the metrics on screen
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(ig_trainer):
            validator.run(val_loader)
            metrics = validator.state.metrics

            final_json[ig_trainer.state.epoch] = metrics

            print(
                f'Validation Results - Epoch[{ig_trainer.state.epoch}] {final_json[ig_trainer.state.epoch]}')

        # Defining that the F1 metric will be responsible for the checkpoints
        def score_function(engine):
            return engine.state.metrics["f1"]

        # Checkpoints configuration
        model_checkpoint = ModelCheckpoint(
            dirname=os.path.join(
                '..', 'output', output_folder_name, self.dataset_name),
            require_empty=False,
            n_saved=1,
            filename_prefix=f'train_{dt_index}',
            score_function=score_function,
            score_name='f1',
            global_step_transform=global_step_from_engine(trainer),
        )

        # Attaching the checkpoint mechanism to each run of the validation
        validator.add_event_handler(
            Events.COMPLETED, model_checkpoint, {"models": model})

        print(
            f'\nTraining the {model.__class__.__name__} models on device: {self.device} (iteration {dt_index})')

        # Running everything for self.epochs
        trainer.run(train_loader, max_epochs=self.epochs)

        print(f'\nTrain finished for models {model.__class__.__name__}')

        # Exporting metrics files
        with open(os.path.join('..', 'output', output_folder_name, self.dataset_name, f'training_results_{dt_index}.json'), 'w') as f:
            json.dump(final_json, f)

        # Saving the training
        model.load_state_dict(load(model_checkpoint.last_checkpoint))

        # Calculating the CAM metrics
        test_dataset_dir = os.path.join(
            '..', 'datasets', f"{self.dataset_name}_{dt_index}", 'test')
        
        get_cam_metrics(model, output_folder_name,
                        self.dataset_name, test_dataset_dir, dt_index)
        
    def procedure_repeated_holdout_test(self, output_folder_name: str, dt_index: int):
        model = self.model

        if self.dataset_name == 'CR':
            self.dataset_name = f'CR_2_CV'
        elif self.dataset_name == 'LA':
            self.dataset_name = f'LA_4_CV'
        elif self.dataset_name == 'LG':
            self.dataset_name = f'LG_2_CV'
        elif self.dataset_name == 'NHL':
            self.dataset_name = f'NHL_3_CV'
        elif self.dataset_name == 'UCSB':
            self.dataset_name = f'UCSB_2_CV'

        model = model.load_state_dict(load(os.path.join('..', 'output', output_folder_name, self.dataset_name, ' '.join(
            f for f in os.listdir(os.path.join('..', 'output', output_folder_name, self.dataset_name)) if f.startswith(f'train_{dt_index}')))))
        
        try:
            os.mkdir(os.path.join('..', 'datasets', f'{self.dataset_name}_{dt_index}', 'tests_with_classes'))
        except OSError as error:
            pass

        test_files = os.listdir(os.path.join('..', 'datasets', f'{self.dataset_name}_{dt_index}', 'test'))

        for test_file in test_files:
            index = test_file.split('_')[0]
            try:
                os.mkdir(os.path.join('..', 'datasets', f'{self.dataset_name}_{dt_index}', 'tests_with_classes', f'{index}'))
            except:
                pass
                
            files_in_tests_with_classes = os.listdir(os.path.join('..', 'datasets', f'{self.dataset_name}_{dt_index}', 'tests_with_classes', f'{index}'))
            if test_file in files_in_tests_with_classes:
                continue
            else:
                shutil.copyfile(os.path.join('..', 'datasets', f'{self.dataset_name}_{dt_index}', 'test', test_file),
                                os.path.join('..', 'datasets', f'{self.dataset_name}_{dt_index}', 'tests_with_classes', f'{index}', test_file))

        test_loader = self.__get_test_loader(
            self.__get_test_dataset(os.path.join('..', 'datasets', f'{self.dataset_name}_{dt_index}', 'tests_with_classes')))

        final_json = {}

        # Pytorch-ignite bit
        val_metrics = {
            "accuracy": Accuracy(),
            "precision": Precision(average='weighted'),
            "recall": Recall(average='weighted'),
            "f1": (Precision(average='weighted') * Recall(average='weighted') * 2 / (
                Precision(average='weighted') + Recall(average='weighted'))),
            "loss": Loss(self.criterion)
        }
        validator = Engine(self.val_step)

        # Attaching metrics
        for name, metric in val_metrics.items():
            metric.attach(validator, name)

        val_bar = ProgressBar(desc="Evaluating...")
        # train_bar.attach(trainer)
        val_bar.attach(validator)

        @validator.on(Events.EPOCH_COMPLETED)
        def log_validation_results(ig_validator):
            metrics = validator.state.metrics

            final_json[ig_validator.state.epoch] = metrics

            print(
                f'Validation Results - Epoch[{ig_validator.state.epoch}] {final_json[ig_validator.state.epoch]}')

        print(
            f'\Getting {self.dataset_name} dataset metrics on iteration {dt_index}')

        # Running everything for self.epochs
        validator.run(test_loader, max_epochs=1)

        # Exporting metrics files
        with open(os.path.join('..', 'output', output_folder_name, self.dataset_name, f'test_results_{dt_index}.json'), 'w') as f:
            json.dump(final_json, f)


    def procedure(self, output_folder_name: str):
        if self.use_repeated_holdout:
            raise TypeError("This procedure should not be user when use_repeated_holdout is enabled.")

        model = self.model
        train_loader, val_loader = self.loaders

        final_json = {}

        # Pytorch-ignite bit
        val_metrics = {
            "accuracy": Accuracy(device=self.device),
            "precision": Precision(average='weighted', device=self.device),
            "recall": Recall(average='weighted', device=self.device),
            "f1": (Precision(average='weighted', device=self.device) * Recall(average='weighted', device=self.device) * 2 / (
                Precision(average='weighted', device=self.device) + Recall(average='weighted', device=self.device))),
            "loss": Loss(self.criterion, device=self.device)
        }

        # Here, we check if a method is actually not overwritten by the class
        trainer = Engine(self.train_step)
        validator = Engine(self.val_step)

        '''
        if len(self.__abstractmethods__) > 0:
            for f in self.__abstractmethods__:
                if f == 'train_step':
                    trainer = create_supervised_trainer(
                        model, self.optimizer, self.criterion, self.device)
                elif f == 'val_step':
                    validator = create_supervised_evaluator(
                        model, val_metrics, self.device)
        '''

        # Attaching metrics
        for name, metric in val_metrics.items():
            metric.attach(validator, name)

        # Creating the loading bards
        train_bar = ProgressBar(desc="Training...")
        val_bar = ProgressBar(desc="Evaluating...")
        train_bar.attach(trainer)
        val_bar.attach(validator)

        # Defining that after each epoch, we should run the validator once and show the metrics on screen
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(ig_trainer):
            validator.run(val_loader)
            metrics = validator.state.metrics

            final_json[ig_trainer.state.epoch] = metrics

            print(
                f'Validation Results - Epoch[{ig_trainer.state.epoch}] {final_json[ig_trainer.state.epoch]}')

        # Defining that the F1 metric will be responsible for the checkpoints
        def score_function(engine):
            return engine.state.metrics["f1"]

        # Checkpoints configuration
        model_checkpoint = ModelCheckpoint(
            dirname=os.path.join(
                '..', 'output', output_folder_name, self.dataset_name),
            require_empty=False,
            n_saved=1,
            filename_prefix=f'train',
            score_function=score_function,
            score_name='f1',
            global_step_transform=global_step_from_engine(trainer),
        )

        # Attaching the checkpoint mechanism to each run of the validation
        validator.add_event_handler(
            Events.COMPLETED, model_checkpoint, {"models": model})

        print(
            f'\nTraining the {model.__class__.__name__} models on device: {self.device}')

        # Running everything for self.epochs
        trainer.run(train_loader, max_epochs=self.epochs)

        print(f'\nTrain finished for models {model.__class__.__name__}')

        # Exporting metrics files
        with open(os.path.join('..', 'output', output_folder_name, self.dataset_name, 'training_results.json'), 'w') as f:
            json.dump(final_json, f)

        # Saving the training
        model.load_state_dict(load(model_checkpoint.last_checkpoint))

        # Calculating the CAM metrics
        test_dataset_dir = os.path.join(
            '..', 'datasets', self.dataset_name, 'test')
        get_cam_metrics(model, output_folder_name,
                        self.dataset_name, test_dataset_dir)

    def new_proceadure(self, output_folder_name: str):
        model = self.model

        if self.dataset_name == 'CR':
            self.dataset_name = 'CR_2_CV'
        elif self.dataset_name == 'LA':
            self.dataset_name = 'LA_4_CV'
        elif self.dataset_name == 'LG':
            self.dataset_name = 'LG_2_CV'
        elif self.dataset_name == 'NHL':
            self.dataset_name = 'NHL_3_CV'
        elif self.dataset_name == 'UCSB':
            self.dataset_name = 'UCSB_2_CV'

        model = model.load_state_dict(load(os.path.join('..', 'output', output_folder_name, self.dataset_name, ' '.join(
            f for f in os.listdir(os.path.join('..', 'output', output_folder_name, self.dataset_name)) if f.endswith('.pt')))))
        test_loader = self.loader

        final_json = {}

        # Pytorch-ignite bit
        val_metrics = {
            "accuracy": Accuracy(),
            "precision": Precision(average='weighted'),
            "recall": Recall(average='weighted'),
            "f1": (Precision(average='weighted') * Recall(average='weighted') * 2 / (
                Precision(average='weighted') + Recall(average='weighted'))),
            "loss": Loss(self.criterion)
        }

        # Here, we check if a method is actually not overwritten by the class
        # trainer = Engine(self.train_step)
        validator = Engine(self.val_step)

        # Attaching metrics
        for name, metric in val_metrics.items():
            metric.attach(validator, name)

        # Creating the loading bards
        # train_bar = ProgressBar(desc="Training...")
        val_bar = ProgressBar(desc="Evaluating...")
        # train_bar.attach(trainer)
        val_bar.attach(validator)

        # Defining that after each epoch, we should run the validator once and show the metrics on screen
        # @trainer.on(Events.EPOCH_COMPLETED)
        @validator.on(Events.EPOCH_COMPLETED)
        def log_validation_results(ig_validator):
            metrics = validator.state.metrics

            final_json[ig_validator.state.epoch] = metrics

            print(
                f'Validation Results - Epoch[{ig_validator.state.epoch}] {final_json[ig_validator.state.epoch]}')

        # # Defining that the F1 metric will be responsible for the checkpoints
        # def score_function(engine):
        #     return engine.state.metrics["f1"]

        # # Checkpoints configuration
        # model_checkpoint = ModelCheckpoint(
        #     dirname=os.path.join(
        #         '..', 'output', output_folder_name, self.dataset_name),
        #     require_empty=False,
        #     n_saved=1,
        #     filename_prefix=f'train',
        #     score_function=score_function,
        #     score_name='f1',
        #     global_step_transform=global_step_from_engine(trainer),
        # )

        # # Attaching the checkpoint mechanism to each run of the validation
        # validator.add_event_handler(
        #     Events.COMPLETED, model_checkpoint, {"models": model})

        print(
            f'\nTraining the {model.__class__.__name__} models on device: {self.device}')

        # Running everything for self.epochs
        validator.run(test_loader, max_epochs=1)

        # print(f'\nTrain finished for models {model.__class__.__name__}')

        # Exporting metrics files
        with open(os.path.join('..', 'output', output_folder_name, self.dataset_name, 'test_results.json'), 'w') as f:
            json.dump(final_json, f)

        # # Saving the training
        # model.load_state_dict(load(model_checkpoint.last_checkpoint))

        # # Calculating the CAM metrics
        # test_dataset_dir = os.path.join(
        #     '..', 'datasets', self.dataset_name, 'test')
        # get_cam_metrics(model, output_folder_name,
        #                 self.dataset_name, test_dataset_dir)
