import math

from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import params


class FashionMNISTDataset(Dataset):
    def __init__(self, csv, transform=None):
        data = pd.read_csv(csv)
        self.X = np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28)
        self.Y = np.array(data.iloc[:, 0])

        del data
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = self.X[idx]
        label = self.Y[idx]

        if self.transform:
            item = self.transform(item)

        return item, label


means = (0.5, 0.5, 0.5)
deviations = means
normalize = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(means, deviations)])

training = FashionMNIST('fashionmnist/', download=True, train=True, transform=normalize)
testing = FashionMNIST('fashionmnist/', download=True, train=False, transform=normalize)

train_size = len(training)
val_size = math.floor(params.val_size * train_size)
train_size = train_size - val_size

training, validating = random_split(training, [train_size, val_size])

print("# of samples:\nTraining: {}k\nValidation: {}k\nTesting: {}k".format(
    len(training)//1000,
    len(validating)//1000,
    len(testing)//1000)
)

train = DataLoader(training, batch_size=params.batch_size, shuffle=True)
val = DataLoader(validating, batch_size=params.batch_size, shuffle=True)
test = DataLoader(testing, batch_size=params.batch_size, shuffle=True)
