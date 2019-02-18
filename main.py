import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import sys

from data import load
import models.fully_connected as fc
import models.cnn_1 as cnn_1
import models.cnn_2 as cnn_2
import models.cnn_3 as cnn_3
import params
import constants

DEFAULT_SHAPE = (-1, constants.im_size)
DEFAULT_MODEL = "cnn_2"
data = load()

def train(model, optimizer, criterion, num_epochs=params.num_epochs, reshape=None, data=data.train, print_every=1000):
    """
    Train a neural model on `data`.

    :param model: an instance of torch.nn.Module, the neural model to train
    :param optimizer: optimizer
    :param criterion: loss function
    :param num_epochs: number of epochs
    :param reshape: how to reshape each batch before training. Must match the expected input shape of `model`
    :param data: a torch.utils.data.DataLoader, the training set
    :param print_every: how often should we print training loss
    :return: a tuple (losses, model, optimizer) with
        - losses: an array containing the loss at each epoch
        - model : the trained model
        - optimizer : optimizer
    """
    losses = []
    num_samples = len(data)
    if reshape is None:
        reshape = DEFAULT_SHAPE
    print("# Training\nStarting training...")
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data):
            images = Variable(images.view(*reshape).float())
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.data.item())

            if (i + 1) % print_every == 0:
                print("Epoch {epoch:d}/{num_epochs:d}\tIter {iter:d}/{num_iter:d}\tLoss {loss:.4f}".format(
                    epoch=epoch+1,
                    num_epochs=num_epochs,
                    iter=i+1,
                    num_iter=num_samples,
                    loss=loss.data.item()
                ))
    print("Training done.\n")
    return losses, model, optimizer

def eval(model, data=data.test, reshape=None):
    """
    Eval `model` on `data`.

    :param model: a trained torch.nn.Module object
    :param data: a torch.utils.data.DataLoader
    :param reshape: optional, a tuple of int. Use it to reshape the data before feeding it to the NN
    :return: accuracy
    """
    if reshape is None:
        reshape = DEFAULT_SHAPE
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in data:
            images = Variable(images.view(*reshape).float())
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    print('# Eval\nTest Accuracy of the model on the {} test images: {:.4f}%\n'.format(total, 100 * correct / total))
    return correct/total


models = {
    "fc": fc,
    "cnn_1": cnn_1,
    "cnn_2": cnn_2,
    "cnn_3": cnn_3
}

def load_model(default=DEFAULT_MODEL):
    if len(sys.argv) > 1 and sys.argv[1] in models:
        default = sys.argv[1]
    nn = models[default]
    print("# Model")
    print("Using model '{}'".format(nn.name))
    print(nn.model)
    print("")
    return nn


if __name__ == "__main__":
    nn = load_model()

    try:
        num_epochs = int(sys.argv[2])
    except (ValueError, IndexError, KeyError):
        num_epochs = params.num_epochs

    losses, trained_model, _ = train(nn.model, nn.optimizer, nn.criterion, reshape=nn.reshape, num_epochs=num_epochs)
    test_acc = eval(trained_model, reshape=nn.reshape)
    val_acc = eval(trained_model, data=data.val, reshape=nn.reshape)
    losses_cnn_in_epochs = losses[0::600]

    plt.xlabel('Epoch #')
    plt.ylabel('Loss for CNN')
    plt.plot(losses_cnn_in_epochs)
    plt.show()
