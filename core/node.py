__all__ = ["Node"]


import torch.nn as nn

from automl.core.operations import Operation


class Node(nn.Module):
    """
        Node class for representing nodes within each cell. Each node takes in
        two inputs, x and y. Operations f and g are applied to x and y,
        respectively, resulting and outputs f(x) and g(y). The final output of
        the node is f(x) + g(y).
    """

    def __init__(
        self,
        x_id,
        x_op,
        y_id,
        y_op,
        x_size,
        y_size,
        channels,
        stride,
    ):
        """
        Arguments:
            x_id: id of the input x.
            x_op: operation to be run on the input x.
            y_id: id of the input y.
            y_op: operation to be run on the input y.
            x_size: size of input x.
            y_size: size of input y.
            channels: number of channels produced by the node.
            stride: stride of operations.
        """
        super().__init__()
        # downsampling only happens when the input is one of the cell's inputs
        x_stride = stride if x_id in [0, 1] else 1
        y_stride = stride if y_id in [0, 1] else 1
        self.x_op = Operation(
            op_idx=x_op,
            input_size=x_size,
            out_channels=channels,
            stride=x_stride
        )
        self.y_op = Operation(
            op_idx=y_op,
            input_size=y_size,
            out_channels=channels,
            stride=y_stride
        )
        # heigth and width of the two outputs must be the same
        assert self.x_op.output_size[1:] == self.y_op.output_size[1:]
        self.output_size = self.x_op.output_size

    def forward(self, x, y):
        x = self.x_op(x)
        y = self.y_op(y)

        return x + y
