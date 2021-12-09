__all__ = ["NASNetwork_TI"]

from itertools import chain

import torch.nn as nn

from automl.core.cell_TI import Cell_TI


class NASNetwork_TI(nn.Module):
    """
    NASNetwork_TI is the child network that is created using the
    architecetures sampled by the controller.

    A block contains num_conv_cells number of convolution cells and one
    reduction cell.
    """

    def __init__(
        self,
        input_size,
        channels,
        num_classes,
        num_blocks,
        num_nodes,
        num_conv_cells,
        dropout
    ):
        """
        Arguments:
            input_size {list of ints} -- size of the input images e.g. [3, 32, 32]
            channels -- number of channels to start with in the stem
            num_classes -- number of classes in dataset
            num_blocks {int} -- number of blocks in the network e.g 3
            num_conv_cells {int} -- number of convolution cells per block e.g. 3
            dropout -- dropout probability e.g. 0
        """
        super().__init__()
        self.conv_arch = None
        self.reduc_arch = None
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_conv_cells = num_conv_cells
        self.dropout = dropout
        # set up stem
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=input_size[0],
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=channels
            )
        )
        # the stem operation increases the channels
        input_size[0] = channels
        output_sizes = [input_size, input_size]
        # set up the cells
        self.cells = nn.ModuleList()
        for block_i in range(self.num_blocks):
            for conv_i in range(self.num_conv_cells):
                conv_cell = Cell_TI(
                    num_nodes=num_nodes,
                    prev_output_sizes=output_sizes,
                    out_channels=channels,
                    reduction=False
                )
                self.cells.append(conv_cell)
                output_sizes = [output_sizes[-1], conv_cell.output_size]
            # double channels every time there's downsampling
            channels *= 2
            reduc_cell = Cell_TI(
                num_nodes=num_nodes,
                prev_output_sizes=output_sizes,
                out_channels=channels,
                reduction=True
            )
            self.cells.append(reduc_cell)
            output_sizes = [output_sizes[-1], reduc_cell.output_size]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(output_sizes[-1][0], self.num_classes)

    def forward(self, x):
        """Forward pass of the child model.

        Arguments:
            x {[type]} -- input to child model. An image e.g. [3, 84, 84]

        Returns:
            [type] -- output of child model
        """
        out0 = out1 = self.stem(x)
        for i in range(len(self.cells)):
            out0, out1 = out1, self.cells[i](out0, out1)
        out = out1
        out = self.avg_pool(out)
        out = self.dropout(out)
        out = self.fc(out.view(out.size(0), -1))

        return out
        
    def update_network(self, conv_arch, reduc_arch):
        """
            Updates the specific child network given the convolution and
            reduction architectures.

            Arguments:
                conv_arch: the convolution architecture in token format
                reduc_arch: the reduction architecture in token format
        """
        self.conv_arch = conv_arch
        self.reduc_arch = reduc_arch
        # Iterate through cells and update using architectures
        for i in range(len(self.cells)):
            if i % self.num_conv_cells == 0 and i != 0:
                self.cells[i].update_cell(self.reduc_arch)
            else:
                self.cells[i].update_cell(self.conv_arch)

    def parameters(self):
        """
            Returns a list of parameters that are specific to the subnetwork as
            defined by the provided convolution and reduction architectures.
        """
        if self.conv_arch is None or self.reduc_arch is None:
            raise ValueError("Convolution and reduction architectures were not specified!")

        parameters = chain.from_iterable([cell.parameters() for cell in self.cells])

        return parameters
