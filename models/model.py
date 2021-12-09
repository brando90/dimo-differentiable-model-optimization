__all__ = ["NASNetwork"]

import torch.nn as nn

from automl.core.cell import Cell
from automl.core.operations import Identity


class NASNetwork(nn.Module):
    """
    NASNetwork is the child network that is created using the
    architecetures sampled by the controller.

    A block contains num_conv_cells number of convolution cells and one
    reduction cell.
    """

    def __init__(
        self,
        archs,
        input_size,
        num_classes,
        num_blocks,
        num_conv_cells,
        in_channels,
        dropout,
    ):
        """
        Arguments:
            archs {list tuples length 4} -- a tuple consisting of convolution and reduction
                architectures represented as a list of tokens. 
                Note: tuples are are (n0 op0 n1 op1), where n0 and n1 are two previous
                    nodes and op0 and op1 are operations to be applied to n0 and n1,
                    espectively. 
                e.g.
                (conv)     0:[(0, 3, 0, 3), (0, 3, 0, 3), (0, 3, 0, 3), (4, 3, 4, 3), (0, 3, 4, 3)]
                (reduc)    1:[(0, 3, 0, 3), (0, 3, 0, 3), (2, 3, 2, 3), (4, 3, 4, 3), (0, 3, 4, 3)]

            input_size {list of ints} -- size of the input images e.g. [3, 32, 32]
            num_classes -- number of classes in dataset
            num_blocks {int} -- number of blocks in the network e.g 3
            num_conv_cells {int} -- number of convolution cells per block e.g. 3
            in_channels {int} -- number of channels in the input
            dropout -- dropout probability e.g. 0
        """
        super().__init__()
        self.conv_arch = archs[0] # e.g. [(0, 3, 0, 3), (0, 3, 0, 3), (0, 3, 0, 3), (4, 3, 4, 3), (0, 3, 4, 3)]
        self.reduc_arch = archs[1] # e.g. [(0, 3, 0, 3), (0, 3, 0, 3), (2, 3, 2, 3), (4, 3, 4, 3), (0, 3, 4, 3)]
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_conv_cells = num_conv_cells
        self.in_channels = in_channels # e.g. 3
        self.dropout = dropout
        self.stem0 = nn.Sequential(Identity(input_size, 1, in_channels)) # what are these for?
        self.stem1 = nn.Sequential(Identity(input_size, 1, in_channels)) # what are these for?
        # input_size for both is image size since stem0 and stem0 are just Identity()
        output_sizes = [input_size, input_size]
        self.cells = nn.ModuleList()
        for block_i in range(self.num_blocks):
            #print(f'-> block_i = {block_i}')
            # generate N conv cells
            for conv_i in range(self.num_conv_cells):
                #print(f'conv_i = {conv_i}')
                #print(f'output_sizes = {output_sizes}')
                #print(f'self.in_channels = {self.in_channels}')
                cell = Cell(
                    self.conv_arch,
                    output_sizes,
                    self.in_channels, 
                    reduction=False
                )
                self.cells.append(cell)
                output_sizes = [output_sizes[-1], cell.output_size]
            # generate single reduction cell
            #print(f'reduct_cell')
            #print(f'output_sizes = {output_sizes}')
            self.in_channels *= 2
            reduct_cell = Cell(self.reduc_arch, output_sizes, self.in_channels, reduction=True)
            self.cells.append(reduct_cell)
            output_sizes = [output_sizes[-1], reduct_cell.output_size]
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
        out0 = self.stem0(x)
        out1 = self.stem1(out0)
        for cell in self.cells:
            out0, out1 = out1, cell(out0, out1)
        out = out1
        out = self.avg_pool(out)
        out = self.dropout(out)
        out = self.fc(out.view(out.size(0), -1))
        return out

if __name__ == "__main__":
    import torch
    # Arguments
    conv = [
        (1, 4, 1, 2),
        (1, 4, 1, 2),
        (1, 4, 0, 4),
        (0, 4, 0, 4),
        (0, 4, 0, 4)
    ]
    reduc = [
        (0, 4, 0, 4),
        (0, 4, 0, 4),
        (0, 4, 0, 4),
        (0, 4, 0, 4),
        (0, 4, 0, 4)
    ]
    archs = (conv, reduc)
    input_size = [3, 32, 32]
    num_classes = 10
    num_blocks = 2
    num_conv_cells = 2
    in_channels = 3
    dropout = 0
    x = torch.rand(1572864).view(512, 3, 32, 32)
    # Model
    nasnetwork = NASNetwork(
        archs=archs,
        input_size=input_size,
        num_classes=num_classes,
        num_blocks=num_blocks,
        num_conv_cells=num_conv_cells,
        in_channels=in_channels,
        dropout=dropout
    )

    out = nasnetwork(x)
    