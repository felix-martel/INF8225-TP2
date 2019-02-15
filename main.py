# TP2 - INF8225
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch

import data
import cnn
import params
import constants

# Datasets
def train(model, optimizer, criterion, num_epochs=params.num_epochs, reshape=(-1, constants.im_size), batch_size=params.batch_size, data=data.train, print_every=1000):
    losses = []
    print("Starting training...")
    # Main loop
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data):
            images = Variable(images.view(reshape).float())
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
                    num_iter=len(data) // batch_size,
                    loss=loss.data[0]
                ))
    print("Training done.")
    return losses, model, optimizer

def eval(model, data=data.test):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in data:
            images = Variable(images.float())
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the {} test images: {:.4f}%'.format(total, 100 * correct / total))
    return correct/total

print("A")
losses, trained_model, _ = train(cnn.model, cnn.optimizer, cnn.criterion, reshape=cnn.reshape)
print("B")
test_acc = eval(trained_model)
print("C")
losses_cnn_in_epochs = losses[0::600]
print("D")

plt.xlabel('Epoch #')
plt.ylabel('Loss for CNN')
plt.plot(losses_cnn_in_epochs)
plt.show()
