import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from utils.torch_utils import set_requires_grad
from predicting_performance.data_point_models.custom_layers import Flatten

mdl = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3)),
            ('relu2', nn.ReLU()),
            ('Flatten', Flatten()),
            ('fc', nn.Linear(in_features=28 * 28 * 2, out_features=1)),
            ('Sigmoid', nn.Sigmoid())
        ]))

class Cifar10_tutorial_net(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class HardcodedSampler:
    '''
    Sampler that returns a hardcoded model for Cifar.
    This is meant as a prototype for model optimization
    '''

    def __init__(self):
        pass

    def sample_model(self, a):
        #a + 0  # a is an input for consistency when the decoder is the real decoder
        # sample
        mdl = Cifar10_tutorial_net()
        # set_requires_grad(False, self.mdl)
        return mdl
