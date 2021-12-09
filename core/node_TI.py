__all__ = ["Node_TI"]

from itertools import chain

import torch.nn as nn

from automl.core.operations_TI import Operation_TI


class Node_TI(nn.Module):
    """
        Node class for representing nodes within each cell. Each node takes in
        two inputs, x and y. Operations f and g are applied to x and y,
        respectively, resulting and outputs f(x) and g(y). The final output of
        the node is f(x) + g(y).
    """

    def __init__(
        self,
        num_prev_nodes,
        channels,
        stride,
    ):
        """
        Arguments:
            num_prev_nodes: number of previous nodes that informs the number of
                            possible input choices.
            channels: number of channels produced by the node.
            stride: stride of operations.
        """
        super().__init__()
        self.x_op = Operation_TI(
            num_prev_nodes=num_prev_nodes,
            out_channels=channels,
            stride=stride
        )
        self.y_op = Operation_TI(
            num_prev_nodes=num_prev_nodes,
            out_channels=channels,
            stride=stride
        )

    def forward(self, x, y):
        x = self.x_op(x)
        y = self.y_op(y)

        return x + y

    def update_node(self, x_id, x_op, y_id, y_op, out_sizes):
        """
            Updates the node with the needed parameters.

            Arguments:
                x_id: id of input x.
                x_op: operation to be applied to input x.
                y_id: id of input y.
                y_op: operation to be applied to input y.
                out_sizes: a list of all pervious nodes' output sizes.
        """
        x_size, y_size = out_sizes[x_id], out_sizes[y_id]
        self.x_op.update_operation(x_id, x_op, x_size)
        self.y_op.update_operation(y_id, y_op, y_size)
        assert self.x_op.output_size[1:] == self.y_op.output_size[1:]

    def parameters(self):
        parameters = chain.from_iterable([self.x_op.parameters(), self.y_op.parameters()])

        return parameters


    @property
    def output_size(self):
        return self.x_op.output_size
