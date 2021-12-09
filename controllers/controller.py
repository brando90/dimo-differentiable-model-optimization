__all__ = ["Controller", "seq_to_arch"]

import torch
import torch.nn as nn

from automl.decoders.decoder import Decoder, SOS_ID

from pdb import set_trace as st

class Controller(nn.Module):
    """
    Controller that generates architectures using the decoder.
    """

    def __init__(
        self, device, num_layers, num_nodes, num_ops, hidden_size, dropout,
    ):
        """
        Arguments:
            num_layers: number of layers in LSTM
            num_nodes: number of nodes in cell including the two input nodes
            num_ops: number of total operations
            hidden_size: number of features in hidden state of Decoder
            dropout: dropout probability
        """
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.num_ops = num_ops
        self.vocab_size = (self.num_nodes - 1) + self.num_ops + 1
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.length = 8 * (self.num_nodes - 2)
        self.decoder = Decoder(
            self.num_layers,
            self.num_nodes,
            self.num_ops,
            self.vocab_size,
            self.hidden_size,
            self.dropout,
            self.length,
        ).to(device)

    def forward(self, x=None, x_thought=None):
        """ Returns a sequential representation of the architecture to be processed by
        a sampleer to generate a child/base network.
        
        Keyword Arguments:
            x {torch_uu.Tensor} -- initial input to decoder (default: {None})
            x_thought {tensor} -- initial arch thought embedding (default: {None})
        
        Returns:
            
            conv {list init tensors} -- sequential representation of layers (indices) e.g. [tensor([[1]]), tensor([[10]]), tensor([[1]]), tensor([[10]]), tensor([[1]]), tensor([[10]]), tensor([[1]]), tensor([[10]]), tensor([[1]]), tensor([[10]]), tensor([[1]]), tensor([[10]]), tensor([[1]]), tensor([[10]]), ...]
            reduc {list init tensors} -- sequential representation of layers (indices) e.g. [tensor([[1]]), tensor([[10]]), tensor([[1]]), tensor([[10]]), tensor([[1]]), tensor([[10]]), tensor([[1]]), tensor([[10]]), tensor([[1]]), tensor([[10]]), tensor([[1]]), tensor([[10]]), tensor([[1]]), tensor([[10]]), ...]
            out, hidden, cell {torch_uu.Tensor} - tensor output of decoder.
        """
        if x is None:
            x = torch.LongTensor([SOS_ID]).view(1, 1).to(self.device)
        seq, out, hidden, cell = self.decoder(x, x_thought)
        conv, reduc = seq[: self.length // 2], seq[self.length // 2:]
        conv, reduc = seq_to_arch(conv, self.num_nodes), seq_to_arch(reduc, self.num_nodes)

        return conv, reduc, out.view(1, 1, -1), hidden, cell


def seq_to_arch(seq, num_nodes):
    """
    Translates given sequential representation of an architecture sampled by
    the controller to an architecture

    Arguments:
        seq: sequential representation of architecture
        num_nodes: number of nodes in cell including the two input nodes

    Returns:
        a list of 4-tuples (n0 op0 n1 op1), where n0 and n1 are two previous
        nodes and op0 and op1 are operations to be applied to n0 and n1,
        respectively. 
        e.g. [(0, 1, 0, 1), (0, 1, 0, 1), (0, 1, 0, 1), (0, 1, 0, 1), (0, 4, 0, 4)]
    """
    arch = []
    for i in range(0, len(seq), 4):
        arch.append(
            (
                seq[i].item() - 1,
                seq[i + 1].item() - (num_nodes - 1) - 1,
                seq[i + 2].item() - 1,
                seq[i + 3].item() - (num_nodes - 1) - 1,
            )
        )
    return arch
