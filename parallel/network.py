import numpy as np
import torch.nn as nn
import torch
import os
import warnings
from torch.optim import SGD, Adam
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F

EPOCHS = 4
LEARNING_RATE = 0.0003
MOMENTUM = 0.45
OPTIMIZER = "Adam"
IMG_PATH_DEFAULT = "../dsprites-dataset/images/"
bad_seeds = {9: 379, 5: 967, 15: 29, 14: 719}


class ShapeConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ShapeConvolutionalNeuralNetwork, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(392, 6),
            nn.ReLU(),
            nn.Linear(6, 2),
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def train_one_epoch(
    batch_size, epoch_index, model, loss_fn, optimizer, training_loader, disable
):
    running_loss = 0.0
    last_loss = 0.0
    correct = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in (pbar := tqdm(enumerate(training_loader), disable=disable)):
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
            pbar.set_postfix(
                epoch=str(epoch_index + 1),
                batch=str(i + 1),
                loss=float(np.round(last_loss, 2)),
                acc=float(np.round(nccorrect, 2)),
            )
            running_loss = 0.0
            correct = 0

    return last_loss


def load_model(name, bias, num_it=0):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ShapeConvolutionalNeuralNetwork()
    model = model.to(device)
    path = "{}_{}_{}.pickle".format(
        name,
        str(bias).replace("0.", "b0i").replace("1.", "b1i"),
        str(num_it),
    )
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        return model
    warnings.warn(
        f"[model not found] model path '{path}' not found, returning random initialized model"
    )
    model.eval()
    return model


def train_network(
    training_loader,
    bias,
    strength,
    name,
    batch_size=128,
    load=False,
    retrain=False,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    optim=OPTIMIZER,
    cuda_num=0,
    disable=True,
    num_it=0,
    seeded=False,
):
    if seeded:
        if num_it in bad_seeds:
            torch.manual_seed(bad_seeds[num_it])
            np.random.seed(bad_seeds[num_it])
        else:
            torch.manual_seed(num_it)
            np.random.seed(num_it)
    model = ShapeConvolutionalNeuralNetwork()
    device = f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    path = "{}_{}_{}.pickle".format(
        name,
        str(bias).replace("0.", "b0i").replace("1.", "b1i"),
        str(num_it),
    )
    if load and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        if not retrain:
            return model
    if not disable:
        print(device)
        print(path)
    loss_fn = nn.CrossEntropyLoss()
    if optim == "Adam":
        optimizer = Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=MOMENTUM)
    best_loss = 100
    avg_loss = 100
    best_epoch = ""
    for epoch in range(epochs):
        print("EPOCH {}:".format(epoch + 1))
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            batch_size, epoch, model, loss_fn, optimizer, training_loader, disable
        )
        print(f"loss epoch: {avg_loss}")
        # save the model's state
        model_path = f"model_{epoch}_{bias}"
        if avg_loss < best_loss:
            best_epoch = model_path
            best_loss = avg_loss
        if avg_loss > 0.69 and epoch < 2 and not seeded:
            print("resetting parameters")
            for block in model.children():
                for layer in block.children():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()  # type: ignore
        torch.save(model.state_dict(), model_path)
        if avg_loss <= 0.15:
            break

    print(
        "best loss: ", best_loss, " last loss: ", avg_loss, " best epoch: ", best_epoch
    )
    if best_loss < avg_loss:
        print("getting older epoch", best_epoch)
        model.load_state_dict(torch.load(best_epoch))
    torch.save(model.state_dict(), path)
    return model


def accuracy(model, loader, disable=True):
    model.eval()
    correct = 0
    with torch.no_grad():
        for item in tqdm(loader, disable=disable):
            data, target = Variable(item[0]), Variable(item[1])
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    return 100.0 * correct / len(loader.dataset)


def accuracy_per_class(model, loader, disable=True):
    model.eval()
    n_classes = 2
    correct = np.zeros(n_classes, dtype=np.int64)
    wrong = np.zeros(n_classes, dtype=np.int64)
    allcorrect = 0
    with torch.no_grad():
        for item in tqdm(loader, disable=disable):
            data, target = Variable(item[0], requires_grad=False), Variable(
                item[1], requires_grad=False
            )
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            preds = output.data.max(dim=1)[1]
            corr = target == preds
            wro = target != preds
            allcorrect += torch.count_nonzero(corr)
            for i in range(n_classes):
                correct[i] += torch.count_nonzero(target[corr] == i)
                wrong[i] += torch.count_nonzero(target[wro] == i)
    assert correct.sum() + wrong.sum() == len(loader.dataset)
    result = (100.0 * correct / (correct + wrong)).tolist()
    return result + [float(100.0 * allcorrect / len(loader.dataset))]
