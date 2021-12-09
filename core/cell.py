__all__ = ["Cell"]

from collections import defaultdict

import torch
import torch.nn as nn
from graphviz import Digraph

from .node import Node
from .operations import FactorizedReduction, ReLUConvBN


class Cell(nn.Module):
    """
    Cell class for representing cells within a network.
    """
    def __init__(
        self,
        arch,
        prev_output_sizes,
        channels,
        reduction
    ):
        """
        Arguments:
            arch {list tuples length 4} -- a list of tokens representing an architecture
                e.g. [(0, 3, 0, 3), (2, 3, 2, 3), (2, 3, 2, 3), (2, 3, 2, 3), (2, 3, 2, 3)]
            prev_output_sizes -- output sizes of two previous cell outputs
            channels -- channels for convolution operations
            reduction -- a boolean indicating whether the cell is a
                       reduction cell or not
        """
        super().__init__()
        self.arch = arch
        self.channels = channels
        self.num_nodes = len(self.arch)
        self.stride = 2 if reduction else 1
        self.nodes = nn.ModuleList()
        used_nodes = defaultdict(int)
        out_sizes = [prev_output_sizes[0], prev_output_sizes[1]]
        # Calibrate sizes of the two inputs
        self.calibrate_size = CalibrateSize(out_sizes, self.channels)
        out_sizes = self.calibrate_size.output_size
        for i in range(self.num_nodes):
            x_id, x_op, y_id, y_op = self.arch[i]
            x_size, y_size = out_sizes[x_id], out_sizes[y_id]
            node = Node(
                x_id,
                x_op,
                y_id,
                y_op,
                x_size,
                y_size,
                self.channels,
                self.stride
            )
            self.nodes.append(node)
            out_sizes.append(node.output_size)
            used_nodes[x_id] += 1
            used_nodes[y_id] += 1
        self.concat_indices = [i for i in range(self.num_nodes + 2)
                               if not used_nodes[i]]
        out_hw = min(size[1] for i, size in enumerate(out_sizes)
                     if i in self.concat_indices)
        # Calibrate all unused node outputs and concatenate them
        self.final_combine = FinalCombine(
            out_sizes, out_hw, self.channels, self.concat_indices
        )
        self.output_size = [
            self.channels * len(self.concat_indices), out_hw, out_hw
        ]

    def forward(self, prev_out0, prev_out1):
        prev_out0, prev_out1 = self.calibrate_size(prev_out0, prev_out1)
        node_outs = [prev_out0, prev_out1]
        for i in range(self.num_nodes):
            x_id, y_id = self.arch[i][0], self.arch[i][2]
            x, y = node_outs[x_id], node_outs[y_id]
            out = self.nodes[i](x, y)
            node_outs.append(out)

        return self.final_combine(node_outs)

    def graph_cell(self):
        dag = Digraph()
        dag.attr("node", shape="rectangle")
        dag.node("input0", "input0")
        dag.node("input1", "input1")

        return dag


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
                    out_channels=channels,
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
                padding=0,
            )
            x_output_size = [channels, x_hw, x_hw]
        if y_channels != channels:
            self.preprocess_y = ReLUConvBN(
                in_channels=y_channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                padding=0,
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


class FinalCombine(nn.Module):
    """
        Combines the all unused outputs of a cell's nodes to get the final output of the cell. 
    """
    def __init__(
        self, out_sizes, out_hw, channels, concat_indices
    ):
        """
        Arguments:
            out_sizes: output sizes of all nodes in cell
            out_hw: output height/width
            channels: final number of channels
            concat_indices: node outputs to concat
        """
        super(FinalCombine, self).__init__()
        self.out_hw = out_hw
        self.channels = channels
        self.concat_indices = concat_indices
        self.ops = nn.ModuleList()
        self.concat_fac_op_dict = {}
        for i in concat_indices:
            hw = out_sizes[i][1]
            if hw > out_hw:
                assert hw == 2 * out_hw and i in [0, 1]
                self.concat_fac_op_dict[i] = len(self.ops)
                self.ops.append(
                    FactorizedReduction(out_sizes[i], channels)
                )

    def forward(self, states, bn_train=False):
        for i in self.concat_indices:
            if i in self.concat_fac_op_dict:
                states[i] = self.ops[self.concat_fac_op_dict[i]](
                    states[i], bn_train
                )
        out = torch.cat([states[i] for i in self.concat_indices], dim=1)

        return out
