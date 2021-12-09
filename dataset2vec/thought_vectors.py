import torch

def get_thought_vec(thought_arch_dim, thought_arch_hp_dimn, trainable=True):
    '''

    :return torch_uu.Tensor x_thought_vec_arch: thought vector over arch (batch, seq_len, dim)
    :return torch_uu.Tensor x_thought_vec_arch_hp: thought vector over arch_hp (batch, seq_len, dim)
    '''
    thought_dim = thought_arch_dim + thought_arch_hp_dimn
    batch, seq_len = 1, 1
    thought_vec = torch.randn(batch, seq_len, thought_dim, requires_grad=trainable)
    x_thought_vec_arch = thought_vec[:,:,:thought_arch_dim]
    x_thought_vec_arch_hp = thought_vec[:,:,thought_arch_dim:]
    return x_thought_vec_arch, x_thought_vec_arch_hp
