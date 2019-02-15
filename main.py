import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch

import data
from models import cnn
import params
import constants

DEFAULT_SHAPE = (-1, constants.im_size)
params.num_epochs = 10

def train(model, optimizer, criterion, num_epochs=params.num_epochs, reshape=None, batch_size=params.batch_size, data=data.train, print_every=1000):
    losses = []
    num_samples = len(data)
    if reshape is None:
        reshape = DEFAULT_SHAPE
    print("Starting training...")
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
                    loss=loss.data[0]
                ))
    print("Training done.")
    return losses, model, optimizer

def eval(model, data=data.test, reshape=None):
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

    print('Test Accuracy of the model on the {} test images: {:.4f}%'.format(total, 100 * correct / total))
    return correct/total

losses, trained_model, _ = train(cnn.model, cnn.optimizer, cnn.criterion, reshape=cnn.reshape)
test_acc = eval(trained_model, reshape=cnn.reshape)
losses_cnn_in_epochs = losses[0::600]

plt.xlabel('Epoch #')
plt.ylabel('Loss for CNN')
plt.plot(losses_cnn_in_epochs)
plt.show()
