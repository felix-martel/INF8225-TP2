from torch import nn
from torch.optim import Adam
import params
import constants as const

from .utils import LayerParams, LayerList


c1 = LayerParams(n_channels=16, kernel_size=5, padding=2)
mp1 = LayerParams(kernel_size=2, stride=2)
net = LayerList(c1, mp1)

class CNN_1(nn.Module):
    def __init__(self):
        super(CNN_1, self).__init__()
        c_out, h_out, w_out = net.output_shape(*const.im_shape)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(const.num_channels, c1.n_channels, kernel_size=c1.kernel_size, padding=c1.padding),
            nn.BatchNorm2d(c1.n_channels),
            nn.ReLU(),
            nn.MaxPool2d(mp1.kernel_size)
        )
        self.fc = nn.Linear(c_out * h_out * w_out, const.num_classes)

    def forward(self, x):
        out = self.conv_layer(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = CNN_1()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=params.learning_rate)
reshape = (-1, const.num_channels, const.im_height, const.im_width)
name = "1-Layer Convolutional Network"