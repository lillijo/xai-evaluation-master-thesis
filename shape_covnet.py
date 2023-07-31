import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch
import os
#from torchvision import transforms, utils
import pickle
import torchvision
from torch.optim import SGD
from tqdm import tqdm 

class ShapeConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ShapeConvolutionalNeuralNetwork, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=9, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=1),
            nn.Conv2d(8, 8, kernel_size=9, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=1),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(288, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def train_one_epoch(batch_size, epoch_index, model, loss_fn, optimizer, training_loader):
    running_loss = 0.
    last_loss = 0.
    correct = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in (pbar := tqdm(enumerate(training_loader))):
        # Every data instance is an input + label pair
        inputs, labels = data
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).sum().item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            nccorrect = correct / (batch_size)
            pbar.set_description('  batch {} correct {}%'.format(i + 1, np.round(nccorrect,2)))
            tb_x = epoch_index * len(training_loader) + i + 1
            running_loss = 0.
            correct = 0

    return last_loss

def train_network(training_loader, batch_size=128,load=False, path="model_dsprites.pickle"):
    model = ShapeConvolutionalNeuralNetwork()   
    if load:
        model.load_state_dict(torch.load(path))
        return model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.02, momentum=0.5)
    epoch_number = 0
    EPOCHS = 3
    best_vloss = 1_000_000.
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(batch_size, epoch_number, model, loss_fn, optimizer,training_loader)   
        # save the model's state
        model_path = 'model_{}'.format(epoch_number)
        torch.save(model.state_dict(), model_path)

        epoch_number += 1
    torch.save(model.state_dict(), path)
    return model