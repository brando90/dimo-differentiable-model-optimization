import torch
import torch.nn as nn

from collections import OrderedDict

from predicting_performance.data_point_models.custom_layers import View, Flatten

from pdb import set_trace as st

class ModelSampler:
    def __init__(self,data_processor):
        '''
        '''
        ##
        self.data_processor = data_processor
        self.vocab = self.data_processor.vocab
        self.V_a,self.V_hp = len(self.data_processor.vocab.architecture_vocab), len(self.data_processor.vocab.hparms_vocab)
        ##
        self.conv2d_shape = (-1, 8, 8)

    def sample_hp_greedily(self, arch_layer_type, a_hp):
        '''
        Given a layer type and hyperparam activations, samples the most likely
        hyperparams for the give layer type.

        :param torch arch_layer_type:
        :param torch.Tensor a_hp: vector of activations for each allowed hp (1,V_hp)
        :return tuple (values,indicies): the values that are sampled greedily and the corresponding indices.
        '''
        empty_values,empty_indices = ([],[])
        if 'EOS' == str(arch_layer_type):
            return empty_values,empty_indices
        if 'torch_uu.nn.modules.activation.' in str(arch_layer_type):
            return empty_values,empty_indices
        ##
        hparms_vocab = self.data_processor.vocab.hparms_vocab
        indices = self.vocab.hp_start_indices[arch_layer_type] # (1st_hp_idx,2nd_hp_idx,...)
        hps = []
        for start_idx,end_idx in indices:
            # print()
            # print(f'start_idx,end_idx = {start_idx,end_idx}')
            ## get the interval for the activations for the hps
            current_a_hp = a_hp[start_idx:end_idx]
            #print(f'current_a_hp = {current_a_hp}')
            ## greedily get most likely hp
            _,max_idx = current_a_hp.topk(1)
            a_hp_idx = start_idx+max_idx
            #print(f'a_hp_idx = {a_hp_idx}')
            hp = hparms_vocab[a_hp_idx]
            #print(f'hp = {hp}')
            ##
            hps.append(hp)
        return hps

    def hard_sample_model_greedily(self,arch,arch_hp):
        '''
        Given a sequence of layers as distributions, sample the most likely model
        according to a greedy procedure. i.e. max of each layer

        :param torch.Tensor arch: arch soft one-hot over arch types (batch_size,seq_len,V_a)
        :param torch.Tensor arch_hp: arch_hp soft one-hot over arch hyperparams options (batch_size,seq_len,V_hp)

        '''
        architecture_vocab = self.data_processor.vocab.architecture_vocab
        hparms_vocab = self.data_processor.vocab.hparms_vocab
        ## EOS
        #EOS = self.data_processor.tokens2arch_indices('EOS')
        ##
        arch_layers, arch_hp_layers = [], []
        nb_layers = arch.size(1)
        for layer in range(nb_layers):
            ## for current layer, get distribution over arch types
            a, a_hp = arch[0,layer,:], arch_hp[0,layer,:]
            ## get most likely arch type
            (_,a_idx) = a.topk(1)
            a_layer = architecture_vocab[a_idx]
            # print(f'layer = {layer}')
            # print(f'a_idx = {self.data_processor.vocab.architecture_vocab[a_idx]}')
            # print(f'a_layer = {a_layer}')
            ## get most likely arch hp type
            hps = self.sample_hp_greedily(a_layer, a_hp)
            ##
            arch_layers.append(a_layer)
            arch_hp_layers.append(hps)
            ##
            if a_layer == 'EOS':
                break
        ##
        # print(f'arch_layers = {arch_layers}')
        # print(f'arch_hp_layers = {arch_hp_layers}')
        return arch_layers, arch_hp_layers

    def hard_sample_model_beam_search(self):
        '''
        Given a sequence of layers as distributions, sample the most likely model
        according to a beam search. TODO: look up if this is neccessary.
        '''
        pass

    def assemble_seq_model(self, input, layers, hps, verbose=False):
        '''
        :param torch.Tensor input: the tensor with the right shape of the initial data point (e.g. CHW 3,32,32 for cifar10)
        :param list layers: list of layer with hps for the model
        '''
        if verbose:
            print('\n -------- ASSEMBLING MODEL ')
            print(f'layers = {layers}')
            print(f'hps = {hps}')
            print()
        ##
        out = input
        nb_layers = len(layers)
        assembled_layers = []
        for i in range(nb_layers):
            current_layer = layers[i]
            current_hp = hps[i]
            if verbose:
                print(f'i = {i}')
                print(f'current_layer = {current_layer}')
                print(f'current_hp = {current_hp}')
            data_is_flat = out.dim() == 2 # (batch_size,#pixels)
            data_is_image = out.dim() == 4 # (batch_size,C,H,W)
            if (current_layer == nn.Linear):
                ## if current data is CHW but next layer is Linear, insert Flatten layer
                if data_is_image:
                    ## CASE: CHW Linear
                    flt = Flatten()
                    out = flt(out)
                    linear = current_layer(in_features=out.numel(),out_features=current_hp[0])
                    out = linear(out)
                    new_layers = [flt,linear]
                else:
                    ## CASE: just Linear
                    linear = current_layer(in_features=out.numel(),out_features=current_hp[0])
                    out = linear(out)
                    new_layers = [linear]
            elif current_layer == nn.Conv2d:
                ## if current data has been flatten already but next is Conv2d, insert View layer
                if data_is_flat:
                    ## CASE: FLAT Conv2d
                    view = View(shape=self.conv2d_shape)
                    out = view(out) # (batch_size,C,H,W)
                    C = out.size(1)
                    out_channels, kernel_size = current_hp[0], current_hp[1]
                    padding = (kernel_size[0]-1)//2
                    conv2d = current_layer(in_channels=C,out_channels=out_channels,kernel_size=kernel_size,padding=padding)
                    out = conv2d(out)
                    new_layers = [view,conv2d]
                else:
                    ## Case: just Conv2d
                    C = out.size(1)
                    out_channels, kernel_size = current_hp[0], current_hp[1]
                    padding = (kernel_size[0]-1)//2
                    conv2d = current_layer(in_channels=C,out_channels=out_channels,kernel_size=kernel_size,padding=padding)
                    out = conv2d(out)
                    new_layers = [conv2d]
            elif 'torch_uu.nn.modules.activation.' in str(current_layer):
                current_layer = current_layer()
                out = current_layer(out)
                new_layers = [current_layer]
            elif current_layer == 'EOS':
                break
            else:
                raise ValueError(f'Layer: {current_layer} not supported')
            ##
            assembled_layers.extend(new_layers)
        ## assemble model
        mdl = nn.Sequential(*assembled_layers)
        out = mdl(input)
        return mdl
