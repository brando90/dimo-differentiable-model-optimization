__all__ = ["Operation_TI"]

from functools import partial
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_OPS = 5


class Operation_TI(nn.Module):
    """
        Operation_TI class is a convenient wrapper class around all the operations.
        It removes the need for directly calling all the different operations and
        uses indices to call operations, which is consistent with the way
        operations are sampled by the decoder. Each operation also calculates the
        output size, which is crucical for the functionalities within Cell.
    """
    def __init__(self, num_prev_nodes, out_channels, stride):
        """
        Args:
            num_prev_nodes: number of previous nodes that informs the number of
                            possible input choices.
            out_channels: number of channels produced by the operation.
            stride: stride of operation.
        """
        super().__init__()
        self.input_op = None
        self.stride = stride
        self.ops = [
            SepConv_TI(num_prev_nodes, out_channels, 3, 1),
            SepConv_TI(num_prev_nodes, out_channels, 5, 2),
            AvgPool2d(kernel_size=3, padding=1),
            MaxPool2d(kernel_size=3, padding=1),
            Identity(stride, out_channels)
        ]

    def forward(self, x):
        out = self.ops[self.input_op](x)

        return out

    def update_operation(self, input_id, input_op, input_size):
        """
            Updates the operation instance with the needed parameters.
        """
        self.stride = self.stride if input_id in [0, 1] else 1
        self.input_op = input_op
        self.ops[input_op].update(input_size, self.stride, input_id)

    def parameters(self):
        """
          Returns an iterable of the parameters of the operation.
        """
        if self.input_op is None:
            raise ValueError("input_op must be specified in order to access parameters!")

        return self.ops[self.input_op].parameters()

    @property
    def output_size(self):
        if self.input_op is None:
            raise ValueError("output_size cannot be determined without input_op")

        return self.ops[self.input_op].output_size

    def __repr__(self):
        return self.op.__repr__()


class SepConv_TI(nn.Module):
    """
        Separable convolution class.
    """

    def __init__(
        self,
        num_prev_nodes,
        out_channels,
        kernel_size,
        padding
    ):
        """
        Arguments:
            num_prev_nodes: number of previous nodes that informs the number of
                            possible input choices.
            out_channels: number of channels produced by the convolution.
            kernel_size: size of the convolving kernel.
            padding: amount of padding added to the input for convolution.
        """
        super(SepConv_TI, self).__init__()
        self.num_prev_nodes = num_prev_nodes
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.input_size = None
        self.input_id = None
        self.stride = None
        self.relu0 = nn.ReLU()
        self.depth0 = None
        self.point0 = None
        self.bn0 = None
        self.relu1 = nn.ReLU()
        self.depth1 = None
        self.point0 = None
        self.bn1 = None

    def forward(self, x, bn_train=True):
        out = self.relu0(x)
        out = F.conv2d(
            input=out,
            weight=self.depth0[self.input_id],
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=self.in_channels
        )
        out = F.conv2d(
            input=out,
            weight=self.point0[self.input_id],
            padding=0
        )
        out = self.bn0(out, self.input_id, bn_train=bn_train)
        out = self.relu0(out)
        out = F.conv2d(
            input=out,
            weight=self.depth1[self.input_id],
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=self.in_channels
        )
        out = F.conv2d(
            input=out,
            weight=self.point1[self.input_id],
            padding=0
        )
        out = self.bn1(out, self.input_id, bn_train=bn_train)

        return out

    def update(self, input_size, stride, input_id):
        self.stride = stride
        self.input_size = input_size
        self.input_id = input_id
        self.init_weights(input_size[0])

    def init_weights(self, in_channels):
        self.depth0 = nn.ParameterList(
            nn.Parameter(torch.Tensor(in_channels, 1, self.kernel_size, self.kernel_size))
            for i in range(self.num_prev_nodes)
        )
        self.point0 = nn.ParameterList(
            nn.Parameter(torch.Tensor(in_channels, in_channels, 1, 1))
            for i in range(self.num_prev_nodes)
        )
        self.bn0 = BatchNorm2d_TI(in_channels, self.num_prev_nodes)
        self.depth1 = nn.ParameterList(
            nn.Parameter(torch.Tensor(in_channels, 1, self.kernel_size, self.kernel_size))
            for i in range(self.num_prev_nodes)
        )
        self.point1 = nn.ParameterList(
            nn.Parameter(torch.Tensor(in_channels, in_channels, 1, 1))
            for i in range(self.num_prev_nodes)
        )
        self.bn1 = BatchNorm2d_TI(in_channels, self.num_prev_nodes)

    def parameters(self):
        if self.input_id is None:
            raise ValueError("input_id must be specified in order to access parameters!")

        parameters = chain([
            self.depth0[self.input_id],
            self.point0[self.input_id],
            *self.bn0.parameters(self.input_id),
            self.depth1[self.input_id],
            self.point1[self.input_id],
            *self.bn1.parameters(self.input_id)
        ])

        return parameters

    @property
    def output_size(self):
        if self.input_size is None:
            raise ValueError("output_size cannot be determined without input_size")

        return [
            self.out_channels,
            self.input_size[1] // self.stride,
            self.input_size[2] // self.stride
        ]


class BatchNorm2d_TI(nn.Module):
    def __init__(self, num_prev_nodes, num_features):
        super(BatchNorm2d_TI, self).__init__()
        self.num_prev_nodes = num_prev_nodes
        self.weight = nn.ParameterList(
            nn.Parameter(torch.Tensor(num_features))
            for _ in range(num_prev_nodes)
        )
        self.bias = nn.ParameterList(
            nn.Parameter(torch.Tensor(num_features))
            for _ in range(num_prev_nodes)
        )
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.reset_all()

    def forward(self, x, x_id, bn_train=False):
        training = self.training
        if bn_train:
            training = True

        return F.batch_norm(
            input=x,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight[x_id],
            bias=self.bias[x_id],
            training=training
        )

    def parameters(self, input_id):
        return chain([self.weight[input_id], self.bias[input_id]])

    def reset_all(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        for i in range(self.num_prev_nodes):
            self.weight[i].data.fill_(1)
            self.bias[i].data.zero_()


class AvgPool2d(nn.Module):
    """
        Custom nn.AvgPool2d to handle output_size compatibility.
    """

    def __init__(
        self, kernel_size, padding, count_include_pad=False
    ):
        """
            Arguments:
                kernel_size: size of the window.
                padding: amount of padding added to the input for pooling.
                count_include_pad: boolean for including padding in calculation.
        """
        super(AvgPool2d, self).__init__()
        self.input_size = None
        self.stride = None
        self.op = partial(
            F.avg_pool2d,
            kernel_size=kernel_size,
            padding=padding,
            count_include_pad=count_include_pad
        )

    def forward(self, x):
        return self.op(input_size=self.input_size, stride=self.stride)(x)

    def update(self, input_size, stride, *args):
        self.stride = stride
        self.input_size = input_size

    @property
    def output_size(self):
        if self.input_size is None:
            raise ValueError("output_size cannot be determined without input_size")

        return [
            self.input_size[0],
            self.input_size[1] // self.stride,
            self.input_size[2] // self.stride
        ]


class MaxPool2d(nn.Module):
    """
        Custom nn.MaxPool2d to handle output_size compatibility.
    """

    def __init__(self, kernel_size, padding):
        """
            Arguments:
                kernel_size: size of the window.
                padding: amount of padding added to the input for pooling.
        """
        super(MaxPool2d, self).__init__()
        self.input_size = None
        self.stride = None
        self.op = partial(
            F.max_pool2d,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x):
        return self.op(stride=self.stride)(x)


    def update(self, input_size, stride, *args):
        self.input_size = input_size
        self.stride = stride

    @property
    def output_size(self):
        if self.input_size is None:
            raise ValueError("output_size cannot be determined without input_size")

        return [
            self.input_size[0],
            self.input_size[1] // self.stride,
            self.input_size[2] // self.stride
        ]


class Identity(nn.Module):
    """
        Identity operation class. In the case of stride > 1, the identity operation
        also applies a factorized reduction to the input.
    """

    def __init__(self, out_channels, bn_train=False):
        """
            Arguments:
                out_channels: number of channels produced by the operation.
                              Used for determining whether there will be a
                              FactorizedReduction
                bn_train: boolean that indicates it's in training or not for
                          batch normalization.
        """
        super(Identity, self).__init__()
        self.input_size = None
        self.factorized_reduction = None
        self.out_channels = out_channels
        self.bn_train = bn_train

    def forward(self, x):
        if self.factorized_reduction is not None:
            return self.factorized_reduction(x, self.bn_train)
        else:
            return x

    def update(self, input_size, stride, *args):
        self.input_size = input_size
        if stride > 1:
            self.factorized_reduction = FactorizedReduction(
                input_size, self.out_channels
            )

    def parameters(self):
        if self.factorized_reduction is not None:
            return self.factorized_reduction.parameters()
        else:
            return []

    @property
    def output_size(self):
        if self.factorized_reduction is not None:
            return self.factorized_reduction.output_size
        else:
            return self.input_size


class FactorizedReduction(nn.Module):
    """
        Reduction of given input to match desired output size. This is used for
        calibrating sizes.
    """
    def __init__(self, input_size, out_channels):
        """
        Arguments:
            input_size: size of the input.
            out_channels: number of channels produced by this operation.
        """
        super(FactorizedReduction, self).__init__()
        self.input_size = input_size
        self.out_channels = out_channels
        self.path0 = nn.Sequential(
            nn.AvgPool2d(1, 2, 0, count_include_pad=False),
            nn.Conv2d(input_size[0], out_channels // 2, 1, bias=False)
        )
        self.path1 = nn.Sequential(
            nn.AvgPool2d(1, 2, 0, count_include_pad=False),
            nn.Conv2d(input_size[0], out_channels // 2, 1, bias=False)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn.train()
        path0 = x
        path1 = F.pad(x, (0, 1, 0, 1), "constant", 0)[:, :, 1:, 1:]
        out = torch.cat([self.path0(path0), self.path1(path1)], dim=1)
        out = self.bn(out)

        return out

    @property
    def output_size(self):
        return [
            self.out_channels, self.input_size[1] // 2, self.input_size[1] // 2
        ]


class ReLUConvBN(nn.Module):
    """
        ReLU, Conv, and BN in sequence. This combination is used for calibrating sizes.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding
    ):
        """
            Arguments:
                in_channels: number of channels in the input.
                out_channels: number of channels produced by the operation.
                kernel_size: size of the convolving kernel.
                stride: stride of the convolution.
                padding: amoutn of padding added to the input for convolution.
        """
        super(ReLUConvBN, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, bn_train=False):
        x = self.relu(x)
        x = self.conv(x)
        if bn_train:
            self.bn.train()
        x = self.bn(x)

        return x

if __name__ == "__main__":
    import numpy as np
    # Arguments
    num_prev_nodes = 5
    out_channels = 64
    kernel_size = 3
    padding = 1
    input_size = [3, 32, 32]
    stride = 1
    input_id = 0
    # Testing SepConv
    op = SepConv_TI(num_prev_nodes, out_channels, kernel_size, padding)
    op.update(input_size, stride, input_id)
    parameters = op.parameters()
    print("is_leaf: {}".format(np.all([param.is_leaf for param in parameters])))
