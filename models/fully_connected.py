from torch import nn
from torch.optim import SGD
import constants as const
import params

dim_in = const.im_height * const.im_width
dim_out = const.num_classes
dim_hidden = 128
dim_hidden2 = 64

model = nn.Sequential(
    nn.Linear(dim_in, dim_hidden),
    nn.ReLU(),
    nn.Linear(dim_hidden, dim_hidden2),
    nn.ReLU(),
    nn.Linear(dim_hidden2, dim_out),
    nn.LogSoftmax(dim=1)
)

criterion = nn.NLLLoss()
optimizer = SGD(model.parameters(), lr=params.learning_rate)
reshape = None
