from types import SimpleNamespace

import torch

DARTS_CONV = [(0, 0, 1, 0), (0, 0, 1, 0), (0, 4, 1, 0), (0, 4, 2, 0)]
DARTS_REDUC = [(0, 3, 1, 3), (2, 4, 1, 3), (0, 3, 2, 4), (2, 4, 1, 3)]

SNAS_CONV = [(0, 0, 1, 0), (0, 4, 1, 0), (0, 4, 0, 4), (0, 4, 1, 0)]
SNAS_REDUC = [(0, 3, 0, 3), (2, 4, 1, 3), (2, 4, 1, 3), (0, 3, 2, 1)]

ENAS_CONV = [(1, 0, 1, 4), (1, 1, 0, 4), (0, 2, 1, 0), (0, 0, 1, 2), (0, 2, 1, 1)]
ENAS_REDUC = [(0, 1, 1, 2), (1, 0, 1, 2), (1, 2, 1, 0), (4, 1, 1, 2), (5, 0, 0, 1)]

def test_sample_arch():
    from automl.samplers.samplers_cell_based import MicroSampler

    ## get device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    ## Create args
    args = SimpleNamespace()
    args.image_size = 32 #TODO FIX
    args.in_channels = 3
    args.input_size = [args.in_channels, args.image_size, args.image_size]
    args.num_classes = 10
    args.num_blocks = 3
    args.num_conv_cells = 3
    ## controller args
    args.num_layers = 1
    args.num_nodes = 7
    args.num_ops = 5
    args.dropout = 0
    ## Create sampler
    sampler = MicroSampler(
        device=device,
        input_size=args.input_size,
        num_classes=args.num_classes,
        num_blocks=args.num_blocks,
        num_conv_cells=args.num_conv_cells,
        in_channels=args.in_channels
    )
    ## Generate the DARTS child model
    args.conv, args.reduc = DARTS_CONV, DARTS_REDUC
    ## Sample child model from arch
    child_model = sampler(args.conv, args.reduc, args.num_nodes, args.dropout).to(device)
    ##
    x = torch.randn([4]+args.input_size)
    out = child_model(x)
    assert(out is not None)

if __name__ == '__main__':
    test_sample_arch()