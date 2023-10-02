import numpy as np
import torch.nn as nn
import torch
import os
from torch import manual_seed
from torch.optim import SGD
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
from biased_dsprites_dataset import get_dataset

SEED = 37
EPOCHS = 4
LEARNING_RATE = 0.05
MOMENTUM = 0.55


class ShapeConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ShapeConvolutionalNeuralNetwork, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(216, 6),
            nn.ReLU(),
            nn.Linear(6, 2),  # only rectangle and ellipse???
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def train_one_epoch(
    batch_size, epoch_index, model, loss_fn, optimizer, training_loader
):
    running_loss = 0.0
    last_loss = 0.0
    correct = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in (pbar := tqdm(enumerate(training_loader))):
        # Every data instance is an input + label pair
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
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
            last_loss = running_loss / 100  # loss per batch
            nccorrect = correct / (batch_size)
            pbar.set_description(
                "  batch {} correct {}% loss {}".format(
                    i + 1, np.round(nccorrect, 2), np.round(last_loss, 2)
                )
            )
            tb_x = epoch_index * len(training_loader) + i + 1
            running_loss = 0.0
            correct = 0

    return last_loss


def train_network(
    training_loader, bias, strength, name, batch_size=128, load=False, retrain=False
):
    #manual_seed(SEED)
    #np.random.seed(SEED)

    model = ShapeConvolutionalNeuralNetwork()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    model = model.to(device)
    path = "{}_{}_{}.pickle".format(
        name, str(bias).replace("0.", ""), str(strength).replace("0.", "")
    )
    print(path)
    if load and os.path.exists(path):
        model.load_state_dict(torch.load(path))
        if not retrain:
            return model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    epoch_number = 0
    best_loss = 100
    avg_loss = 100
    best_epoch = ""
    for epoch in range(EPOCHS):
        print("EPOCH {}:".format(epoch_number + 1))
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            batch_size, epoch_number, model, loss_fn, optimizer, training_loader
        )
        print(f"loss epoch: {avg_loss}")
        if avg_loss < best_loss:
            best_epoch = f'model_{epoch_number}'
            best_loss = avg_loss
        # save the model's state
        model_path = "model_{}".format(epoch_number)
        torch.save(model.state_dict(), model_path)

        epoch_number += 1
    print("best loss: ", best_loss, " last loss: ", avg_loss, " best epoch: ", best_epoch)
    if best_loss < avg_loss:
        print("getting older epoch", best_epoch)
        model.load_state_dict(torch.load(best_epoch))
    torch.save(model.state_dict(), path)
    return model


def accuracy(model, loader):
    model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for item in tqdm(loader):
            data, target = Variable(item[0]), Variable(item[1])
            output = model(data)
            losses.append(F.cross_entropy(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    eval_loss = float(np.mean(losses))
    return eval_loss, 100.0 * correct / len(loader.dataset)


def accuracy_per_class(model, loader):
    model.eval()
    n_classes = 2
    correct = np.zeros(n_classes, dtype=np.int64)
    wrong = np.zeros(n_classes, dtype=np.int64)
    with torch.no_grad():
        for item in tqdm(loader):
            data, target = Variable(item[0], requires_grad=False), Variable(
                item[1], requires_grad=False
            )
            output = model(data)
            preds = output.data.max(dim=1)[1].cpu().numpy().astype(np.int64)
            t = target.numpy()
            corr = t == preds
            wro = t != preds
            for i in range(n_classes):
                correct[i] += np.count_nonzero(t[corr] == i)
                wrong[i] += np.count_nonzero(t[wro] == i)
    assert correct.sum() + wrong.sum() == len(loader.dataset)
    return 100.0 * correct / (correct + wrong)


def performance_analysis(model, bias, strength):
    train_ds, train_loader, test_ds, test_loader = get_dataset(bias, strength)
    acc_per_class = accuracy_per_class(model, test_loader)
    acc = accuracy(model, test_loader)

    def to_per(arr):
        return ", ".join([f"{np.round(i, 3)}%" for i in arr])

    print(
        f"Biased test dataset accuracy per class: {to_per(acc_per_class)} total accuracy:  {to_per(acc)}% "
    )
    return acc, acc_per_class
