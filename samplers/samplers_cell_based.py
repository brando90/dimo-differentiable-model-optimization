__all__ = ["MicroSampler"]

from automl.controllers.controller import seq_to_arch
from automl.models.model import NASNetwork

import torch.nn as nn

class MicroSampler(nn.Module):
    def __init__(
        self,
        device,
        input_size,
        num_classes,
        num_blocks,
        num_conv_cells,
        in_channels
    ):
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_conv_cells = num_conv_cells
        self.in_channels = in_channels

    def __call__(self, conv, reduc, num_nodes, dropout):
        # convert sequence of tokens to arch format
        archs = (conv, reduc)
        ##
        child_net = NASNetwork(
            archs=archs,
            input_size=self.input_size,
            num_classes=self.num_classes,
            num_blocks=self.num_blocks,
            num_conv_cells=self.num_conv_cells,
            in_channels=self.in_channels,
            dropout=dropout
        ).to(self.device)
        return child_net
    
    def to(self, device):
        '''
        This sampler is not trainable so doing the to method should do nothing
        '''
        return self
