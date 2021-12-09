import torch
import torch.nn as nn

import unittest

import automl.utils.utils_datasets as utils

from predicting_performance.data_processor import DataProcessor
from automl.vocab_full_automl import AutoMLVocab
from automl.samplers.model_sampler import ModelSampler
from automl.decoders.simple_arch_decoder import ArchDecoder
from automl.dataset2vec.thought_vectors import get_thought_vec


class TestHardSampler(unittest.TestCase):

    def test_imports_through_packages(self):
        helloworld = utils.HelloWorld()
        self.assertTrue( helloworld == 'HelloWorld')

    def test_sample_net_from_decoder_distribution(self):
        '''
        Samples a model from the decoder, assembles a model using the sampler
        and tries to pass dummy cifar10 image data through it.
        '''
        ##
        verbose = False
        ## vocab
        vocab = AutoMLVocab()
        data_processor = DataProcessor(vocab)
        V_a, V_hp = len(vocab.architecture_vocab), len(vocab.hparms_vocab)
        ## mdl sampler
        mdl_sampler = ModelSampler(data_processor)
        ## start running tests
        nb_tests = 50
        for i in range(nb_tests):
            ## generate initial thought
            thought_arch_dim, thought_arch_hp_dimn = 64, 128
            x_thought_vec_arch, x_thought_vec_arch_hp = get_thought_vec(thought_arch_dim, thought_arch_hp_dimn)
            ## make decoder
            arch_input_size, arch_hp_input_size = V_a, V_a+V_hp
            arch_hidden_size, arch_hp_hidden_size = 64, 128
            arch_output_size, arch_hp_output_size = V_a, V_hp,
            decoder = ArchDecoder(arch_input_size,arch_hp_input_size,arch_hidden_size,arch_hp_hidden_size,arch_output_size,arch_hp_output_size,data_processor)
            ## output soft model
            arch,arch_hp = decoder(x_thought_vec_arch,x_thought_vec_arch_hp)
            ## sample model layers (from decoder predictions)
            arch_layers, arch_hp_layers = mdl_sampler.hard_sample_model_greedily(arch,arch_hp)
            #print(arch_layers)
            self.assertTrue( arch_layers.count('EOS') <= 1 )
            ## assemble model from model layuers
            batch_size = 1
            CHW = (batch_size,3,32,32) # cifar CHW
            input = torch.randn(*CHW)
            mdl = mdl_sampler.assemble_seq_model(input, arch_layers, arch_hp_layers, verbose=verbose)
            ## forward pass through fake data
            out = mdl(input)
            self.assertTrue(type(out) is torch.Tensor)
        print(f'nb_tests = {nb_tests}')

    def test_assemble_FC_V_Conv2d(self):
        '''
        Test the case where the linear layer has destroyed the locality structure of data.
        This will mean we need to insert a flatten later after the linear layer.
        In addition we need to insert a view layer for the Conv2d which is the code
        we are trying to test.
        '''
        ##
        verbose = False
        ## vocab
        vocab = AutoMLVocab()
        data_processor = DataProcessor(vocab)
        V_a, V_hp = len(vocab.architecture_vocab), len(vocab.hparms_vocab)
        ## mdl sampler
        mdl_sampler = ModelSampler(data_processor)
        ## fake input data
        batch_size = 1
        CHW = (batch_size,3,32,32) # cifar CHW
        input = torch.randn(*CHW)
        ## crate hardcoded model
        arch_layers = [nn.Linear, nn.Conv2d, nn.ReLU]
        arch_hp_layers = [[64], # Linear HPs = (out_features)
                        [16,(3,3)], # Conv2d HPs = (out_channels,kernel_size),
                        ([],[]) # no hps for Activations (ReLU)
                        ]
        ##
        mdl = mdl_sampler.assemble_seq_model(input, arch_layers, arch_hp_layers, verbose=verbose)
        ##
        out = mdl(input)
        self.assertTrue(type(out) is torch.Tensor)

    # def test_full_automl(self):
    #     '''
    #     '''
    #     ##
    #     verbose = False
    #     ## vocab
    #     vocab = AutoMLVocab()
    #     data_processor = DataProcessor(vocab)
    #     V_a, V_hp = len(vocab.architecture_vocab), len(vocab.hparms_vocab)
    #     ## mdl sampler
    #     mdl_sampler = ModelSampler(data_processor)
    #     ## generate initial thought
    #     thought_arch_dim, thought_arch_hp_dimn = 64, 128
    #     x_thought_vec_arch, x_thought_vec_arch_hp = get_thought_vec(thought_arch_dim, thought_arch_hp_dimn)
    #     ## make decoder
    #     arch_input_size, arch_hp_input_size = V_a, V_a+V_hp
    #     arch_hidden_size, arch_hp_hidden_size = 64, 128
    #     arch_output_size, arch_hp_output_size = V_a, V_hp,
    #     decoder = ArchDecoder(arch_input_size,arch_hp_input_size,arch_hidden_size,arch_hp_hidden_size,arch_output_size,arch_hp_output_size,data_processor)
    #     ## sample model layers (from decoder predictions)
    #     arch,arch_hp = decoder(x_thought_vec_arch,x_thought_vec_arch_hp) # output soft model
    #     arch_layers, arch_hp_layers = mdl_sampler.hard_sample_model_greedily(arch,arch_hp)
    #     ## assemble model from model layuers
    #     batch_size = 1
    #     CHW = (batch_size,3,32,32) # cifar CHW
    #     input = torch_uu.randn(*CHW)
    #     mdl = mdl_sampler.assemble_seq_model(input, arch_layers, arch_hp_layers, verbose=verbose)
    #     ## forward pass through fake data
    #     out = mdl(input)
    #     self.assertTrue(type(out) is torch_uu.Tensor)


if __name__ == '__main__':
    unittest.main()
