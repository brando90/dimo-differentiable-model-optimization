__all__ = ["Operation"]

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_OPS = 5


class Operation(nn.Module):
    """
        Operation class is a convenient wrapper class around all the operations.
        It removes the need for directly calling all the different operations and
        uses indices to call operations, which is consistent with the way
        operations are sampled by the decoder. Each operation also calculates the
        output size, which is crucical for the functionalities within Cell.
    """
    def __init__(self, op_idx, input_size, out_channels, stride):
        """
        Args:
            op_idx: index of the selected operation.
            input_size: size of the input in CHW format.
            out_channels: number of channels produced by the operation.
            stride: stride of operation.
        """
        super().__init__()
        self.op_idx = op_idx
        self.ops = [
            SepConv(input_size, out_channels, 3, stride, 1),
            SepConv(input_size, out_channels, 5, stride, 2),
            AvgPool2d(input_size, kernel_size=3, stride=stride, padding=1),
            MaxPool2d(input_size, kernel_size=3, stride=stride, padding=1),
            Identity(input_size, stride, out_channels),
        ]
        self.op = self.ops[self.op_idx]
        self.output_size = self.op.output_size

    def forward(self, x):
        out = self.op(x)
        return out

    def __repr__(self):
        return self.op.__repr__()


class SepConv(nn.Module):
    """
        Separable convolution class.
    """

    def __init__(
        self,
        input_size,
        out_channels,
        kernel_size,
        stride,
        padding
    ):
        """
        Arguments:
            input_size: size of the input.
            out_channels: number of channels produced by the convolution.
            kernel_size: size of the convolving kernel.
            stride: stride of the convolution.
            padding: amount of padding added to the input for convolution.
        """
        super(SepConv, self).__init__()
        self.input_size = input_size
        self.out_channels = out_channels
        self.stride = stride
        in_channels = input_size[0]
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.op(x)

    @property
    def output_size(self):
        return [
            self.out_channels,
            self.input_size[1] // self.stride,
            self.input_size[2] // self.stride
        ]


class AvgPool2d(nn.AvgPool2d):
    """
        Custom nn.AvgPool2d to handle output_size compatibility.
    """

    def __init__(
        self, input_size, kernel_size, stride, padding, count_include_pad=False
    ):
        """
            Arguments:
                input_size: size of the input.
                kernel_size: size of the window.
                stride: stride of the window.
                padding: amount of padding added to the input for pooling.
                count_include_pad: boolean for including padding in calculation.
        """
        super(AvgPool2d, self).__init__(
            kernel_size, stride, padding, count_include_pad=count_include_pad
        )
        self.input_size = input_size
        self.stride = stride

    @property
    def output_size(self):
        return [
            self.input_size[0],
            self.input_size[1] // self.stride,
            self.input_size[2] // self.stride
        ]


class MaxPool2d(nn.MaxPool2d):
    """
        Custom nn.MaxPool2d to handle output_size compatibility.
    """

    def __init__(self, input_size, kernel_size, stride, padding):
        """
            Arguments:
                input_size: size of the input.
                kernel_size: size of the window.
                stride: stride of the window.
                padding: amount of padding added to the input for pooling.
        """
        super(MaxPool2d, self).__init__(kernel_size, stride, padding)
        self.input_size = input_size
        self.stride = stride

    @property
    def output_size(self):
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

    def __init__(self, input_size, stride, out_channels, bn_train=False):
        """
            Arguments:
                input_size: size of the input.
                stride: stride of the operation. Used for determining whether
                        there will be a FactorizedReduction.
                out_channels: number of channels produced by the operation.
                              Used for determining whether there will be a
                              FactorizedReduction
                bn_train: boolean that indicates it's in training or not for
                          batch normalization.
        """
        super(Identity, self).__init__()
        self.input_size = input_size
        self.bn_train = bn_train
        if stride > 1:
            self.factorized_reduction = FactorizedReduction(
                input_size, out_channels
            )
        else:
            self.factorized_reduction = None

    def forward(self, x):
        if self.factorized_reduction is not None:
            return self.factorized_reduction(x, self.bn_train)
        else:
            return x

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


class SPP(nn.Module):
    def __init__(self, channels, levels):
        super(SPP, self).__init__()
        self.channels = channels
        self.levels = levels

    def forward(self, x):
        batch_size, height, width = x.size(0), x.size(2), x.size(2)
        out_list = []
        for level in self.levels:
            h_kernel = int(math.ceil(height / level))
            w_kernel = int(math.ceil(width / level))
            h_pad0 = int(math.floor((h_kernel * level - height) / 2))
            h_pad1 = int(math.ceil((h_kernel * level - height) / 2))
            w_pad0 = int(math.floor((w_kernel * level - width) / 2))
            w_pad1 = int(math.ceil((w_kernel * level - width) / 2))
            pool = nn.MaxPool2d(
                kernel_size=(h_kernel, w_kernel),
                stride=(h_kernel, w_kernel),
                padding=(0, 0)
            )
            padded_x = F.pad(x, pad=[w_pad0, w_pad1, h_pad0, h_pad1])
            out = pool(padded_x)
            print(out.size())
            out_list.append(out.view(batch_size, -1))

        return torch.cat(out_list, dim=1)

    @property
    def output_size(self):
        output_size = 0
        for level in self.levels:
            output_size += self.channels * level * level

        return output_size


if __name__ == "__main__":
    # Arguments
    input_size = [3, 32, 32]
    in_channels = 3
    out_channels = 32
    stride = 1
    affine = True
    x = torch.rand([3072]).view(input_size)

    # Testing operations output_size
    op0 = Operation(
        op_idx=0,
        input_size=input_size,
        out_channels=out_channels,
        stride=stride,
    )
    op1 = Operation(
        op_idx=1,
        input_size=input_size,
        out_channels=out_channels,
        stride=stride,
    )
    op2 = Operation(
        op_idx=2,
        input_size=input_size,
        out_channels=out_channels,
        stride=stride,
    )
    op3 = Operation(
        op_idx=3,
        input_size=input_size,
        out_channels=out_channels,
        stride=stride,
    )
    op4 = Operation(
        op_idx=4,
        input_size=input_size,
        out_channels=out_channels,
        stride=stride,
    )
    print(op0.output_size)
    print(op1.output_size)
    print(op2.output_size)
    print(op3.output_size)
    print(op4.output_size)
