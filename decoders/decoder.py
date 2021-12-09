# NAO code: https://github.com/renqianluo/NAO

__all__ = ["Decoder"]

import torch.nn as nn
import torch.nn.functional as F

SOS_ID = 0


class Decoder(nn.Module):
    """Decoder module for generating Cell architecture.

    Decoder class that uses a multi-layered LSTM to generate Cell architectures.
    For a given node, the decoder samples two nodes and two operations. In order
    to do so, the LSTM does so by greedy decoding or beam search with k = 1,
    i.e. recursively decoding by generating an output and feeding that output as
    the next input.
    """

    def __init__(
        self,
        num_layers,
        num_nodes,
        num_ops,
        vocab_size,
        hidden_size,
        dropout,
        length
     ):
        """
        Arguments:
            num_layers: number of layers in LSTM
            num_nodes: number of nodes in cell including the two input nodes
            num_ops: number of total operations
            vocab_size: number of unique vocabs = (num_nodes - 1) + num_ops + 1
            hidden_size: number of features in hidden state of LSTM
            dropout: dropout probability
            length: total length of the sequential representation of cell
        """
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.num_ops = num_ops
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.length = length
        self.rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.hidden_size
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(
            in_features=self.hidden_size, out_features=self.vocab_size
        )

    def forward(self, x, x_thought=None):
        """
        Argument:
            x: initial input to the LSTM, which is the SOS_ID Tensor

        Return:
            a list of tokens that represent a cell architecture
        """
        self.batch_size = x.size(0)
        seq = []
        hidden, cell = self._init_states()
        if x_thought is not None:
            hidden = x_thought
        for i in range(self.length):
            out, hidden, cell = self._step(x, hidden, cell)
            token = self._decode(i, out)
            seq.append(token)
            x = token
        return seq, out, hidden, cell

    def _step(self, x, hidden, cell):
        """ A single LSTM step.

        Applies the LSTM to a single-element input sequence. Since the decoding
        is done in a naive greedy search manner, the input sequence must be
        single-element. Afterwards, the resulting output is run through a
        nn.Linear layer to output a vector of size vocab_size.

        Arguments:
            x: sequential input to the LSTM
            hidden: the current hidden state of the LSTM
            cell: the current cell state of the LSTM

        Returns:
            out: final output after nn.Linear and log_softmax
            hidden: the hidden state of the LSTM
            cell: the cell state of the LSTM
        """
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        out, (hidden, cell) = self.rnn(input=embedded, hx=(hidden, cell))
        out = self.fc(out.contiguous().view(-1, self.hidden_size))
        out = F.log_softmax(out, dim=1)
        return out, hidden, cell 

    def _decode(self, step, out):
        """ Decodes the output to retrieve node / operation.

        Decodes the output of the LSTM to retrieve the node or operation.
        For each node, the decoder must make four decisions:
            1) sample the first previous node (A)
            2) sample the operation to be performed on node (A)
            3) sample the second previous node (B)
            4) sample the operation to be performed on node (B)

        Given that the possibility of previous nodes increases as the
        decoder progresses to the later nodes, the decoding step must
        dynamically accommodate for the increasing choices of previous
        nodes. This is done by using the current step in the overall
        decoding process to limit the argmax of the decoder output to
        a specific range.

        [  0,     1 ... num_nodes - 1,   num_nodes ... num_nodes + num_ops - 1]
         SOS_ID,       NODE_ID,                       OP_ID

        Arguments:
            step: current step in the decoding of the sequence
            out: output of the current step

        Returns:
            a token that represents the sampled node or operation in the
            current step
        """
        # out = self.fc(out.contiguous().view(-1, self.hidden_size))
        # out = F.log_softmax(out, dim=1)
        if step % 2:
            # sampling index of operation
            token = out[:, self.num_nodes:].topk(1)[1] + self.num_nodes
        else:
            # sampling index of node
            node_end = step // 2 % 10 // 2 + 3
            token = out[:, 1:node_end].topk(1)[1] + 1

        return token

    def _init_states(self):
        """Initializes hidden state and cell state.

        Initialies the hidden state and cell state of the LSTM to zero.
        Refer to link below for the way the initialization is done.
        https://discuss.pytorch.org/t/correct-way-to-declare-hidden-and-cell-states-of-lstm/15745

        TODO: link does not work anymore

        Returns:
            two zero Tensors for the hidden state and the cell state of the LSTM
        """
        param_data = next(self.rnn.parameters()).data
        hidden = param_data.new(
            self.num_layers, self.batch_size, self.hidden_size
        )
        hidden.requires_grad = True
        cell = param_data.new(
            self.num_layers, self.batch_size, self.hidden_size
        )
        cell.requires_grad = True

        return hidden, cell
