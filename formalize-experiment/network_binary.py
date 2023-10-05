import numpy as np
import torch.nn as nn
import torch
import os
from torch import manual_seed
from torch.optim import SGD, Adam
from tqdm import tqdm
from torch.autograd import Variable

SEED = 37
EPOCHS = 4
LEARNING_RATE = 0.0003
OPTIMIZER = "Adam"


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
            nn.Linear(6, 1),  # only rectangle and ellipse???
            nn.Sigmoid(),
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
        # print(outputs.shape, inputs.shape, labels.shape)
        # Compute the loss and its gradients
        labels = labels.data.view_as(outputs).float()
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        # pred = outputs.data.max(1, keepdim=True)[1]
        # print progress
        acc = (outputs.round() == labels).float().mean()
        if i % 100 == 99:
            last_loss = running_loss / 100  # loss per batch
            pbar.set_postfix(
                epoch=str(epoch_index + 1),
                batch=str(i + 1),
                loss=float(last_loss),
                acc=float(acc),
            )
            running_loss = 0.0

    return last_loss


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
):
    manual_seed(SEED)
    np.random.seed(SEED)

    model = ShapeConvolutionalNeuralNetwork()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    model = model.to(device)
    path = "{}_{}_{}_{}.pickle".format(
        name,
        str(bias).replace("0.", "b0i"),
        str(strength).replace("0.", "s0i"),
        str(learning_rate).replace("0.", "l0i"),
    )
    print(path)
    if load and os.path.exists(path):
        model.load_state_dict(torch.load(path))
        if not retrain:
            return model
    loss_fn = nn.BCELoss()  # binary cross entropy
    if optim == "Adam":
        optimizer = Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    best_loss = 100
    avg_loss = 100
    best_epoch = ""
    for epoch in range(epochs):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            batch_size, epoch, model, loss_fn, optimizer, training_loader
        )
        print(f"loss epoch: {avg_loss}")
        # save the model's state
        model_path = "model_{}".format(epoch)
        if avg_loss < best_loss:
            best_epoch = model_path
            best_loss = avg_loss
        torch.save(model.state_dict(), model_path)

    print(
        "best loss: ", best_loss, " last loss: ", avg_loss, " best epoch: ", best_epoch
    )
    if best_loss < avg_loss:
        print("getting older epoch", best_epoch)
        model.load_state_dict(torch.load(best_epoch))
    torch.save(model.state_dict(), path)
    return model


def accuracy(model, loader):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for item in tqdm(loader):
            data, target = Variable(item[0]), Variable(item[1])
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            output = output.data.view_as(target).float()
            acc = (output.round() == target).float().mean()
            total_acc += acc
    return 100.0 * total_acc / len(loader)


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
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            output = output.data.view_as(target).float()
            corr = output.round() == target
            wro = output.round() != target
            for i in range(n_classes):
                correct[i] += torch.count_nonzero(target[corr] == i)
                wrong[i] += torch.count_nonzero(target[wro] == i)
    assert correct.sum() + wrong.sum() == len(loader.dataset)
    return 100.0 * correct / (correct + wrong)
