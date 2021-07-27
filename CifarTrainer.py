import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

from Models import *

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("message.log", 'w')])


class CifarPytorchTrainer:

    def __init__(self, model_name: str, epochs: int, lr: float, batch_size: int, saving_directory: str):
        """
        Implement training on CIFAR dataset and evaluate for testing set

        Args:
            model_name: one of the models architecture from MODELS dict
            epochs: number of epoch for model training
            lr: learning rate for optimizer
            batch_size: batch_size for training process
            saving_directory: location where have to save final model # ./best_model/best_model.pt
        """

        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.saving_directory = saving_directory
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = MODELS[self.model_name]()
        self.model.to(self.device)
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.metrics = {}
        self.valid_size = 0.2

    def get_data(self):
        num_workers = 0

        train_data = datasets.CIFAR10('data', train=True, download=True, transform=self.transforms)
        test_data = datasets.CIFAR10('data', train=False, download=True, transform=self.transforms)

        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=train_sampler,
                                       num_workers=num_workers)
        self.valid_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=valid_sampler,
                                       num_workers=num_workers)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, num_workers=num_workers)

    def train(self):
        """training process for cifar dataset
        """
        valid_loss_min = np.Inf
        for epoch in range(1, self.epochs + 1):
            train_loss = 0.0
            valid_loss = 0.0

            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                # forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                # backward pass:
                loss.backward()
                self.optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)

            # validate the model
            self.model.eval()
            for batch_idx, (data, target) in enumerate(self.valid_loader):
                data, target = data.to(self.device), target.to(self.device)
                # forward pass:
                output = self.model(data)
                loss = self.criterion(output, target)
                # update average validation loss
                valid_loss += loss.item() * data.size(0)

            # calculate average losses
            train_loss = train_loss / len(self.train_loader.sampler)
            valid_loss = valid_loss / len(self.valid_loader.sampler)

            # print training/validation statistics
            logging.info('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

            if valid_loss <= valid_loss_min:
                valid_loss_min = valid_loss
            elif (valid_loss - valid_loss_min) > 0.01:
                break

    def infer(self, new_image: np.ndarray) -> np.ndarray:
        """makes an inference on a single image and returns a probability of each class

        Args:
            new_image: single image

        Returns: a probability of each class

        """
        self.model.eval()
        new_image = new_image.reshape(new_image.shape[1], new_image.shape[1], new_image.shape[0])
        new_image = self.transforms(new_image)
        new_image = new_image.reshape(1, new_image.shape[0], new_image.shape[1], new_image.shape[1])
        new_image = new_image.to(self.device)
        output = self.model(new_image)
        prob = F.softmax(output).detach().cpu().numpy()

        return prob

    def get_metrics(self, test_loader: DataLoader) -> dict:
        """ calculate main metrics on test_loader
        Args:
            test_loader: data for which evaluation metrics should be calculated

        Returns: metrics on given data

        """
        labels = torch.zeros(0).to(self.device)
        predictions = torch.zeros(0).to(self.device)

        self.model.eval()
        # iterate over test data
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            # forward pass
            output = self.model(data)
            _, predicted = torch.max(output, 1)
            labels = torch.cat([labels, target])
            predictions = torch.cat([predictions, predicted])

        labels = labels.cpu().numpy()
        predictions = predictions.cpu().numpy()

        self.metrics['accuracy'] = accuracy_score(labels, predictions)
        self.metrics['f1'] = f1_score(labels, predictions, average='macro')
        self.metrics['precision'] = precision_score(labels, predictions, average='macro')
        self.metrics['recall'] = recall_score(labels, predictions, average='macro')
        self.metrics['balanced_accuracy'] = balanced_accuracy_score(labels, predictions)

        logging.info('Test Balanced Accuracy (Overall): {}'.format(100 * self.metrics['balanced_accuracy']))

        return self.metrics

    def save(self):
        """

        Returns: save a model weights and metrics (as a JSON file)

        """

        with open(self.saving_directory + 'metrics.json', 'w') as f:
            json.dump(self.metrics, f)

        state_dict = {}
        for key in self.model.state_dict():
            state_dict[key] = self.model.state_dict()[key].tolist()

        with open(self.saving_directory + 'results.json', 'w') as f:
            json.dump(state_dict, f)


