import torch
import torch.nn as nn

from torch.nn.functional import softmax as softmax

from predicting_performance.data_processor import DataProcessor
from automl.vocab_full_automl import AutoMLVocab
from automl.samplers.model_sampler import ModelSampler

from automl.utils.torch_utils import get_init_hidden

from pdb import set_trace as st

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class ArchDecoder(nn.Module):

    def __init__(self, arch_input_size, arch_hp_input_size,
                arch_hidden_size, arch_hp_hidden_size,
                arch_output_size, arch_hp_output_size,
                data_processor,
                summary_activation=torch.nn.ReLU(),
                arch_num_layers=1, arch_hp_num_layers=1, bidirectional=False, max_arch_depth=12):
        '''

        TODO: biderectional

        Note:
            - input to lstm: input of shape (seq_len, batch, input_size)
        '''
        super().__init__()
        ##
        self.data_processor = data_processor
        self.vocab = self.data_processor.vocab
        self.V_a,self.V_hp = len(data_processor.vocab.architecture_vocab), len(data_processor.vocab.hparms_vocab)
        ## LSTM unit for processing the architecture
        batch_first = True #If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        self.arch = nn.LSTM(input_size=arch_input_size,
                            hidden_size=arch_hidden_size,
                            num_layers=arch_num_layers,
                            batch_first=batch_first)
        self.arch_out = nn.Linear(arch_hidden_size, arch_output_size)
        ## LSTM unit for processing the architecture hp
        self.arch_hp = nn.LSTM(input_size=arch_hp_input_size,
                            hidden_size=arch_hp_hidden_size,
                            num_layers=arch_hp_num_layers,
                            batch_first=batch_first)
        self.hp_summarizer = nn.Linear(arch_hp_hidden_size, arch_hp_hidden_size//2)
        self.arch_hp_out = nn.Linear(arch_hp_hidden_size, arch_hp_output_size)
        self.summary_activation = summary_activation
        ##
        self.max_arch_depth = max_arch_depth

    def forward(self, x_thought_vec_arch, x_thought_vec_arch_hp):
        '''
        Takes in a thought vectors x and produces a distribution over the architecture and its hypterparameters.

        TODO:
            - output initialization algorithm too...? or even learn that fully...
            - do we need variable length sequences here?
            - increase it to be batch size > 1?
            - How do we processes many thought vectors in a batch but token by token?

        :return torch_uu.Tensor arch: arch soft one-hot over arch types (batch_size,seq_len,V_a)
        :return torch_uu.Tensor arch_hp: arch_hp soft one-hot over arch hyperparams options (batch_size,seq_len,V_hp)
        '''
        V_a, V_hp = self.V_a, self.V_hp
        ## data params
        batch_size = x_thought_vec_arch.size(0)
        seq_len = x_thought_vec_arch.size(1)
        ## EOS
        #EOS = self.data_processor.tokens2arch_indices('EOS')
        ## initialize LSTMs memory and cell
        _, c_a = get_init_hidden(batch_size, self.arch.hidden_size, self.arch.num_layers, self.arch.bidirectional)
        _, c_hp = get_init_hidden(batch_size, self.arch_hp.hidden_size, self.arch_hp.num_layers, self.arch.bidirectional)
        h_a, h_hp = x_thought_vec_arch, x_thought_vec_arch_hp
        ## SOS
        SOS_a, SOS_a_hp = torch.ones(batch_size,seq_len,V_a)/V_a, torch.ones(batch_size,seq_len,V_hp)/V_hp
        a, a_hp = SOS_a, SOS_a_hp # (batch_size,seq_len,V_a), (batch_size,seq_len,V_hp)
        ## initial values of arch
        #a_idx, a_hp_idx = -1, -1
        # arch, arch_indices = [], []
        # arch_hp, arch_hp_indices = [], []
        arch = []
        arch_hp = []
        for i in range(self.max_arch_depth):
            # print(f'i = {i}')
            # print(f'a_idx = {self.data_processor.vocab.architecture_vocab[a_idx]}')
            # print(f'a = {a}')
            # print(f'a.size() = {a.size()}')
            # if a_idx == EOS:
            #     break
            ## get soft representation of arch
            a, h_a, c_a = self.forward_arch(a, h_a, c_a)
            a_hp, h_hp, c_hp = self.forward_arch_hp(a, h_a, a_hp, h_hp, c_hp)
            ## add layers to lists
            #arch_indices.append(a_idx), arch_hp_indices.append(a_hp_idx)
            arch.append(a), arch_hp.append(a_hp)
        ## convert the lists to a sequential model tensor
        arch = torch.cat(arch,dim=1)
        arch_hp = torch.cat(arch_hp,dim=1)
        return arch,arch_hp

    def forward_arch(self, a, h_a, c_a):
        '''
        :param torch.Tensor a: a is the arch feature-vec (or one-hot), (batch_size,seq_len,V_a)
        '''
        a, (h_a, c_a) = self.arch(input=a, hx=(h_a, c_a)) # lstm
        a = softmax( self.arch_out(a), dim=2 )
        return a, h_a, c_a

    def forward_arch_hp(self, a, h_a, a_hp, h_hp, c_hp):
        '''

        Q: is it worth while to also pass the a_hp of the previous layer?
        '''
        ## compression layer: compressed_h_hp = g(h_a, h_hp*W_comp)
        h_hp = self.summary_activation( self.hp_summarizer(h_hp) )
        prev_h_hp = torch.cat( (h_a, h_hp), dim=2 )
        ## lstm layer
        input = torch.cat( (a, a_hp), dim=2 )
        a_hp, (h_hp, c_hp) = self.arch_hp(input=input, hx=(c_hp, prev_h_hp)) # lstm
        ## out-layer
        a_hp = self.arch_hp_out(a_hp)
        a_hp = softmax(a_hp, dim=2)
        return a_hp, h_hp, c_hp
