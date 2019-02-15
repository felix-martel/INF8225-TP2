#TP2 - INF8225

#Libraries
import torch
import numpy as np
import torchvision
import pandas as pd
import matplotlib.pyplot as plt 

#Froms
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets



#Variables
num_epochs = 20
batch_size = 64
learning_rate = 0.001

#Dataset Definition for CNN
class FashionMNISTDataset(Dataset):
    def __init__(self, csv, transform=None):
        data = pd.read_csv(csv)
        self.X = np.array(data.iloc[:,1:]).reshape(-1,1,28,28)
        self.Y = np.array(data.iloc[:,0])

        del data
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        item  = self.X[idx]
        label = self.Y[idx]

        if self.transform :
            item = self.transform(item)
        
        return (item,label)


#Datasets
train_dataset = FashionMNISTDataset(csv='fashionmnist/fashion-mnist_train.csv')
test_dataset  = FashionMNISTDataset(csv='fashionmnist/fashion-mnist_test.csv')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(7*7*32,10)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr = learning_rate)

losses_cnn = []

#Main loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images.float())
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses_cnn.append(loss.data.item())
        
        # Print every 1000 step
        if (i+1) % 1000 == 0:
            print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))


cnn.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.float())
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))


losses_cnn_in_epochs = losses_cnn[0::600]


dim_in = 28*28
dim_out = 10
dim_hidden = 128
dim_hidden2 = 64


means = (0.5, 0.5, 0.5)
deviations = means
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(means, deviations)])


training = datasets.FashionMNIST('fashionmnist/', download=True, train=True, transform=transform)

training_batches = torch.utils.data.DataLoader(training, batch_size=batch_size, shuffle=True)

testing = datasets.FashionMNIST('fashionmnist/',download=True, train=False, transform=transform)

test_batches = torch.utils.data.DataLoader(testing, batch_size=batch_size, shuffle=True)                                               

model = nn.Sequential(
    nn.Linear(dim_in, dim_hidden),
    nn.ReLU(),
    nn.Linear(dim_hidden, dim_hidden2),
    nn.ReLU(),
    nn.Linear(dim_hidden2,dim_out),
    nn.LogSoftmax(dim=1)
)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    running_loss = 0
    for images, labels in training_batches:

        images = Variable(images.view(images.shape[0], -1).float())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if not epoch % 10:
        print("Training loss: {running_loss/len(data_batches)}")


correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_batches:
        images = images.view(images.shape[0], -1)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

plt.xlabel('Epoch #')
plt.ylabel('Loss for CNN')
plt.plot(losses_cnn_in_epochs)
plt.show()
