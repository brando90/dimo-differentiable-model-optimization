__all__ = ["Cell_TI"]

from collections import defaultdict
from itertools import chain

import torch
import torch.nn as nn

from automl.core.node_TI import Node_TI
from .operations import FactorizedReduction, ReLUConvBN


class Cell_TI(nn.Module):
    """
        Cell class for representing cells within a network.
    """
    def __init__(
        self,
        num_nodes,
        prev_output_sizes,
        out_channels,
        reduction
    ):
        """
        Arguments:
            num_nodes -- number of nodes in the cell, not including the input nodes
            prev_output_sizes -- sizes of two immediately preceding outputs
            out_channels -- number of channels for convolution operations output
            reduction -- a boolean indicating whether the cell is a reduction
                         cell or not
        """
        super().__init__()
        self.arch = None
        self.final_combine = None
        self.num_nodes = num_nodes
        self.out_channels = out_channels
        self.stride = 2 if reduction else 1
        self.out_sizes = [prev_output_sizes[0], prev_output_sizes[1]]
        # the two inputs might need to be calibrated in terms of size
        self.calibrate_size = CalibrateSize(self.out_sizes, out_channels)
        self.out_sizes = self.calibrate_size.output_size
        # generate all possible connections within the cell
        self.nodes = []
        for curr_node in range(2, self.num_nodes + 2): # idx 0 and 1 are input nodes
            node = Node_TI(
                num_prev_nodes=curr_node,
                channels=self.out_channels,
                stride=self.stride
            )
            self.nodes.append(node)

        if reduction:
            self.factorized_reduction0 = FactorizedReduction(
                self.out_sizes[0], self.out_channels
            )
            self.factorized_reduction1 = FactorizedReduction(
                self.out_sizes[1], self.out_channels
            )

    def update_cell(self, arch):
        """
            Updates the architecture of the cell. This should be called whenever
            a new architecture is sampled by the controller.

            Argument:
                arch: the new architecture of the cell.
        """
        self.arch = arch
        for i, (x_id, x_op, y_id, y_op) in enumerate(arch):
            self.nodes[i].update_node(x_id, x_op, y_id, y_op, self.out_sizes)
            self.out_sizes.append(self.nodes[i].output_size)

        self.setup_final_combine()

    def parameters(self):
        """
            Returns an iterable of the parameters of the cell that is selected
            by the architecture.

            Returns: an iterable containing all the parameters in the cell that
                     is specified by provided architecture.
        """
        if self.arch is None:
            raise ValueError("Architecture must be specified in order to access parameters!")

        parameters = chain.from_iterable(
            [self.nodes[i].parameters() for i in range(self.num_nodes)]
        )

        return parameters

    def forward(self, prev_out0, prev_out1):
        """
            Arguments:
                prev_out0: output of the cell that is two layers before
                prev_out1: output of the cell that is one layer before
        """
        prev_out0, prev_out1 = self.calibrate_size(prev_out0, prev_out1)
        node_outs = [prev_out0, prev_out1]
        for i in range(self.num_nodes):
            x_id, _, y_id, _ = self.arch[i]
            x, y = node_outs[x_id], node_outs[y_id]
            out = self.nodes[i](x, y)
            node_outs.append(out)

        if self.reduction:
            if 0 in self.concat_indices:
                node_outs[0] = self.factorized_reduction0(node_outs[0])
            if 1 in self.concat_indices:
                node_outs[1] = self.factorized_reduction1(node_outs[1])

        out = torch.cat([node_outs[i] for i in range(self.concat_indices)], dim=1)

        return self.final_combine(out)

    def setup_final_combine(self):
        """
            Uses the provided architecture, whether it is convolution or
            reduction, to combine any unused node outputs within the cell
            using FinalCombine.
        """
        used_nodes = defaultdict(int)
        for x_id, _, y_id, _ in self.arch:
            used_nodes[x_id] += 1
            used_nodes[y_id] += 1
        self.concat_indices = [i for i in range(self.num_nodes + 2)
                               if not used_nodes[i]]
        self.final_combine = ReLUConvBN(
            in_channels=len(self.concat_indices) * self.out_sizes[-1][0],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    @property
    def output_size(self):
        return [
            self.out_channels,
            self.out_sizes[1][-1] // self.stride,
            self.out_sizes[1][-1] // self.stride
        ]

class CalibrateSize(nn.Module):
    """
        Calibrates the sizes of the two inputs of a cell to match a desired output size.
    """
    def __init__(self, output_sizes, channels):
        super().__init__()
        self.channels = channels
        x_hw, y_hw = output_sizes[0][-1], output_sizes[1][-1]
        x_channels, y_channels = output_sizes[0][0], output_sizes[1][0]
        x_output_size = [x_channels, x_hw, x_hw]
        y_output_size = [y_channels, y_hw, y_hw]
        # input0 is a conv and input1 is a reduc
        if x_hw != y_hw:
            assert x_hw == y_hw * 2
            self.preprocess_x = nn.Sequential(
                nn.ReLU(),
                FactorizedReduction(
                    input_size=output_sizes[0],
                    out_channels=channels
                )
            )
            x_output_size = [channels, y_hw, y_hw]
        # input0 and input1 are both conv
        elif x_channels != channels:
            self.preprocess_x = ReLUConvBN(
                in_channels=x_channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
            x_output_size = [channels, x_hw, x_hw]
        if y_channels != channels:
            self.preprocess_y = ReLUConvBN(
                in_channels=y_channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
            y_output_size = [channels, y_hw, y_hw]
        self.output_size = [x_output_size, y_output_size]

    def forward(self, x, y, bn_train=False):
        if x.size(-1) != y.size(-1):
            x = self.preprocess_x(x)
        elif x.size(1) != self.channels:
            x = self.preprocess_x(x, bn_train=bn_train)
        if y.size(1) != self.channels:
            y = self.preprocess_y(y, bn_train=bn_train)

        return [x, y]
