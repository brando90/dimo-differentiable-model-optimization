#from __future__ import division, print_function, absolute_import

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch import sigmoid, cat

import numpy as np

import higher
from higher.optim import DifferentiableOptimizer

from collections import OrderedDict
#import math

from uutils.torch_uu import preprocess_grad_loss, get_stats

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class MyEmptyOptimizer(Optimizer):

    def __init__(self, params, trainable_opt_model, trainable_opt_state, *args, **kwargs):
        defaults = {
            'trainable_opt_model':trainable_opt_model, 
            'trainable_opt_state':trainable_opt_state, 
            'args':args, 
            'kwargs':kwargs
        }
        super().__init__(params, defaults)

class TrainableSGD(DifferentiableOptimizer):

    def _update(self, grouped_grads, **kwargs):
        prev_lr = self.param_groups[0]['trainable_opt_state']['prev_lr']
        eta = self.param_groups[0]['trainable_opt_model']['eta']
        # start differentiable & trainable update
        zipped = zip(self.param_groups, grouped_grads)
        lr = 0.01*eta(prev_lr).view(1)
        fg = 1
        for group_idx, (group, grads) in enumerate(zipped):
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue
                p_new = p - lr*g
                group['params'][p_idx] = p_new
        # fake returns
        self.param_groups[0]['trainable_opt_state']['prev_lr'] = lr

class Eta1D(nn.Module):

    def __init__(self, device=None, bias=True):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.eta = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(1,1, bias=True)),
            ('sigmoid', nn.Sigmoid())
        ])).to(device)
        self.device = device

    def forward(self, prev_lr):
        lr = self.eta(prev_lr)
        return lr

    def get_trainable_opt_state(self, out, h, c, *args, **kwargs):
        prev_lr = ((out.mean()+h.mean()+c.mean())/3).view(1)
        trainable_opt_state = {'prev_lr': prev_lr}
        return trainable_opt_state

higher.register_optim(MyEmptyOptimizer, TrainableSGD)

class EmptyLstmTrainableLR(Optimizer):

    def __init__(self, params, trainable_opt_model, trainable_opt_state):
        defaults = {'trainable_opt_model':trainable_opt_model, 'trainable_opt_state':trainable_opt_state}
        super().__init__(params, defaults)

class LstmTrainableLR(DifferentiableOptimizer):
    '''
    Trainable learning rate. All the parameters have the same (trainable) learning rate 
    for the whole model. So it's not a unique trainable learning rate per parameter as
    in meta-lstm (in Optimization as a model for few shot learning).
    '''

    def _update(self, grouped_grads, **kwargs):
        out, h, c = self.param_groups[0]['trainable_opt_state']['prev_lstm_state']
        eta = self.param_groups[0]['trainable_opt_model']
        # start differentiable & trainable update
        zipped = zip(self.param_groups, grouped_grads)
        lr, out_eta, h, c = eta(out, h, c)
        for group_idx, (group, grads) in enumerate(zipped):
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue
                p_new = p - lr*g
                group['params'][p_idx] = p_new
        # fake returns
        self.param_groups[0]['trainable_opt_state']['prev_lstm_state'] = (out_eta, h, c)

class LstmEta(nn.Module):

    def __init__(self, device, input_size, hidden_size, num_layers, D=1):
        super().__init__()
        # gpu
        self.device = device
        # trainable scheduler (optimizer params)
        self.lstm_eta = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # TODO: learn a step-size per layer
        self.eta = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(in_features=hidden_size, out_features=D)),
            ('sigmoid0', nn.Sigmoid())
        ]))
        self.to(device)

    def forward(self, out_etat, ht, ct):
        out_eta, (h, c) = self.lstm_eta(input=out_etat, hx=(ht, ct))
        eta = self.eta(input=out_eta)
        eta = eta.view(1)
        return eta, out_eta, h, c

    def get_trainable_opt_state(self, out, h, c, *args, **kwargs):
        # this is because the out has to be size 64 because the input to the 
        # inner opt is the previous out of the previous learning rate.
        # this first one doesn't matter that much but it's still correlated to out of the decoder
        # because of out.mean(). 
        # Again what matters is that the input to the the next state is out of the prev
        # state from the *inner optimizer*, the first one doesn't matter too much.
        #out = (h.detach() + c.detach() + out.mean())/3
        out = out.mean().expand_as(h)
        trainable_opt_state = {'prev_lstm_state': (out, h, c)}
        return trainable_opt_state

higher.register_optim(EmptyLstmTrainableLR, LstmTrainableLR)

class EmptyLstmTrainableLRFG(Optimizer):

    def __init__(self, params, trainable_opt_model, trainable_opt_state):
        defaults = {'trainable_opt_model':trainable_opt_model, 'trainable_opt_state':trainable_opt_state}
        super().__init__(params, defaults)

class LstmTrainableLRFG(DifferentiableOptimizer):
    '''
    Trainable learning rate. All the parameters have the same (trainable) learning rate and trainable forgate gate
    for the whole model. So it's not a unique trainable learning rate per parameter as
    in meta-lstm (in Optimization as a model for few shot learning).
    '''

    def _update(self, grouped_grads, **kwargs):
        (out_lr, h_lr, c_lr) = self.param_groups[0]['trainable_opt_state']['prev_lstm_lr_state']
        (out_fg, h_fg, c_fg) = self.param_groups[0]['trainable_opt_state']['prev_lstm_fg_state']
        eta_lr_fg = self.param_groups[0]['trainable_opt_model']
        # start differentiable & trainable update
        zipped = zip(self.param_groups, grouped_grads)
        (lr, out_lr, h_lr, c_lr), (fg, out_fg, h_fg, c_fg) = eta_lr_fg(out_lr, h_lr, c_lr, out_fg, h_fg, c_fg)
        for group_idx, (group, grads) in enumerate(zipped):
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue
                p_new = fg*p - lr*g
                group['params'][p_idx] = p_new
        # fake returns
        self.param_groups[0]['trainable_opt_state']['prev_lstm_lr_state'] = (out_lr, h_lr, c_lr)
        self.param_groups[0]['trainable_opt_state']['prev_lstm_fg_state'] = (out_fg, h_fg, c_fg)

class LstmEtaLRFG(nn.Module):

    def __init__(self, device, input_size, hidden_size, num_layers, D=1):
        super().__init__()
        # gpu
        self.device = device
        # trainable Learning Rate (LR)
        self.lstm_lr = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.eta_lr = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(in_features=hidden_size, out_features=D)),
            ('sigmoid0', nn.Sigmoid())
        ]))
        # trainable Forget Gate (FG)
        self.lstm_fg = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.eta_fg = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(in_features=hidden_size, out_features=D)),
            ('sigmoid0', nn.Sigmoid())
        ]))
        self.to(device)

    def forward(self, out_lrt, h_lrt, c_lrt, out_fgt, h_fgt, c_fgt):
        # compute Learning Rate (LR)
        out_lr, (h_lr, c_lr) = self.lstm_lr(input=out_lrt, hx=(h_lrt, c_lrt))
        lr = self.eta_lr(input=out_lr)
        lr = lr.view(1)
        # Compute Forget Gate (FG)
        out_fg, (h_fg, c_fg) = self.lstm_fg(input=out_fgt, hx=(h_fgt, c_fgt))
        fg = self.eta_fg(input=out_fg)
        fg = fg.view(1)
        return (lr, out_lr, h_lr, c_lr), (fg, out_fg, h_fg, c_fg)

    def get_trainable_opt_state(self, out, h, c, *args, **kwargs):
        # this is because the out has to be size 64 because the input to the 
        # inner opt is the previous out of the previous learning rate.
        # this first one doesn't matter that much but it's still correlated to out of the decoder
        # because of out.mean(). 
        # Again what matters is that the input to the the next state is out of the prev
        # state from the *inner optimizer*, the first one doesn't matter too much.
        #out = (h.detach() + c.detach() + out.mean())/3
        out = out.mean().expand_as(h)
        trainable_opt_state = {}
        # initial state for learnable learning rate and forget rate are the prev out, h, c 
        # from the previous LSTM
        trainable_opt_state['prev_lstm_lr_state'] = (out, h, c)
        trainable_opt_state['prev_lstm_fg_state'] = (out, h, c)
        return trainable_opt_state

higher.register_optim(EmptyLstmTrainableLRFG, LstmTrainableLRFG)

####
####

class EmptyLstmTrainableLRFG_LandscapeAware(Optimizer):

    def __init__(self, params, *args, **kwargs):
        defaults = {
            'args':args, 
            'kwargs':kwargs
        }
        super().__init__(params, defaults)

class LstmTrainableLRFG_LandscapeAware(DifferentiableOptimizer):
    '''
    Trainable learning rate. All the parameters have the same (trainable) learning rate and trainable forgate gate
    for the whole model. So it's not a unique trainable learning rate per parameter as
    in meta-lstm (in Optimization as a model for few shot learning).
    '''

    def _update(self, grouped_grads, **kwargs):
        ## unpack params to update
        trainable_opt_model = self.param_groups[0]['kwargs']['trainable_opt_model']
        trainable_opt_state = self.param_groups[0]['kwargs']['trainable_opt_state']
        lr, fg, h, c = trainable_opt_state['prev_state']
        inner_train_loss = self.param_groups[0]['kwargs']['trainfo_kwargs']['inner_train_loss']
        inner_train_err = self.param_groups[0]['kwargs']['trainfo_kwargs']['inner_train_err']
        # get the flatten params & grads
        zipped = zip(self.param_groups, grouped_grads)
        flatten_params = []
        flatten_grads = []
        for group_idx, (group, grads) in enumerate(zipped):
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue
                g = g.view(-1) # flatten vector
                p = p.view(-1) # flatten vector
                assert( len(g.size())==1 and len(p.size())==1 )
                flatten_grads.append(g)
                flatten_params.append(p)
        flatten_params = torch.cat(flatten_params, dim=0)
        flatten_grads = torch.cat(flatten_grads, dim=0)
        ## get param & grad stats
        param_stats = get_stats(flatten_tensor=flatten_params) # [mu, std, min_v, max_v, med]
        grad_stats = get_stats(flatten_tensor=flatten_grads) # [mu, std, min_v, max_v, med]
        # pre-process grads
        grad_stats_prep = []
        for g_stat in grad_stats:
            if len(g_stat.size()) == 0:
                g_stat = g_stat.unsqueeze(0)
            g_stat_prep = preprocess_grad_loss(g_stat)
            grad_stats_prep.append(g_stat_prep)
        grad_stats_prep = torch.stack(grad_stats_prep).view(-1)
        param_stats = torch.stack(param_stats).view(-1)
        # get global lr & fg (landscape aware)
        lr, fg, h, c = trainable_opt_model(
            prep_grad=grad_stats_prep, 
            loss=inner_train_loss, err=inner_train_err, 
            ht=h, ct=c, 
            param_stats=param_stats, 
            lrt=lr, fgt=fg)
        # do update
        zipped = zip(self.param_groups, grouped_grads) # needed because 1st iterator is consumed
        for group_idx, (group, grads) in enumerate(zipped):
            #print(f'group_idx = {group_idx}')
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue
                p_new = fg*p - lr*g
                group['params'][p_idx] = p_new
        # fake returns
        self.param_groups[0]['kwargs']['trainable_opt_state']['prev_state'] = [lr, fg, h, c]

class LstmEtaLRFG_LandscapeAware(nn.Module):

    def __init__(self, device, input_size, hidden_size, num_layers, D=1):
        super().__init__()
        # gpu
        self.device = device
        # lstim for feature extraction for training
        # h = lstm(prep_grad, loss, err)
        self.lstm_features_for_training = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # trainable Learning Rate (LR)
        # i^<t> = fc(h, theta, i^<t-1>] + b_I) = lr([lstm(prep_grad, loss, err), theta, i^<t-1>] + b_I)
        self.eta_lr = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(in_features=hidden_size+5+1, out_features=D)), # 5 for params stats, 1 for lr^<t-1>
            ('sigmoid0', nn.Sigmoid())
        ]))
        # trainable Forget Gate (FG)
        # f^<t> = fc(h, theta, f^<t-1>] + b_I) = lr([lstm(prep_grad, loss, err), theta, f^<t-1>] + b_F)
        self.eta_fg = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(in_features=hidden_size+5+1, out_features=D)), # 5 for params stats, 1 for f^<t-1>
            ('sigmoid0', nn.Sigmoid())
        ]))
        self.to(device)

    def forward(self, prep_grad, loss, err, ht, ct, param_stats, lrt, fgt):
        ## compute feature space: h = lstm(prerp_grad, loss, err)
        xt_lstm = torch.cat([prep_grad,loss.view(1),err.view(1)]).view(1,1,-1)
        out,(h,c) = self.lstm_features_for_training(input=xt_lstm, hx=(ht,ct))
        ## compute Learning Rate (LR): lr^<t> = NN(h,theta,lr^<t-1>)
        xt_lr = torch.cat([h.squeeze(),param_stats,lrt]).view(1,-1)
        lr = self.eta_lr(input=xt_lr)
        lr = lr.view(1)
        # Compute Forget Gate (FG): f^<t> = NN(h,theta,f^<t-1>)
        xt_fg = torch.cat([h.squeeze(),param_stats,fgt]).view(1,-1)
        fg = self.eta_fg(input=xt_fg)
        fg = fg.view(1)
        return lr, fg, h, c

    def get_trainable_opt_state(self, out, h, c, *args, **kwargs):
        inner_opt = kwargs['inner_opt']
        # process hidden state from arch decoder/controller
        h = ( ( out.mean()+h )/2 ).expand_as(h)
        # initial lr & fg as input to the trainable-optimizer
        lr, fg = torch.torch.Tensor([0.0]).view(1).to(device), torch.Tensor([0.0]).view(1).to(device)
        trainable_opt_state = {}
        if inner_opt is None: # inner optimizer has not been used yet
            #lr, fg = torch_uu.torch_uu.Tensor([0.0]).view(1).to(device), torch_uu.Tensor([0.95]).view(1).to(device)
            pass
        else: 
            ## Uncomment to use info from a the inner optimizer from previous outer loop TODO: probably shpuld be using a f;ag with args.
            #[lr_opt, fg_opt, h_opt, c_opt] = inner_opt.param_groups[0]['kwargs']['trainable_opt_state']['prev_state']
            # if you want to use the prev inner optimizer's h & c
            #h, c = (h + h_opt.detach())/2, (c + c_opt.detach())/2
            #lr, fg = lr_opt.detach(), fg_opt.detach()
            pass
        trainable_opt_state['prev_state'] = [lr, fg, h, c]
        ## create initial trainable opt state
        return trainable_opt_state

higher.register_optim(EmptyLstmTrainableLRFG_LandscapeAware, LstmTrainableLRFG_LandscapeAware)

####
####

class EmptyMetaLstmOptimizer(Optimizer):

    def __init__(self, params, *args, **kwargs):
        defaults = {
            'args':args, 
            'kwargs':kwargs
        }
        super().__init__(params, defaults)

class MetaTrainableLstmOptimizer(DifferentiableOptimizer):
    '''
    Adapted lstm-meta trainer from Optimization as a model for few shot learning.
    '''

    def _update(self, grouped_grads, **kwargs):
        ## unpack params to update
        trainable_opt_model = self.param_groups[0]['kwargs']['trainable_opt_model']
        trainable_opt_state = self.param_groups[0]['kwargs']['trainable_opt_state']
        [(lstmh, lstmc), metalstm_hx] = trainable_opt_state['prev_state']
        inner_train_loss = self.param_groups[0]['kwargs']['trainfo_kwargs']['inner_train_loss']
        inner_train_err = self.param_groups[0]['kwargs']['trainfo_kwargs']['inner_train_err']
        # get the flatten params & grads
        zipped = zip(self.param_groups, grouped_grads)
        flatten_params = []
        flatten_grads = []
        flatten_lengths = []
        original_sizes = []
        for group_idx, (group, grads) in enumerate(zipped):
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue
                original_sizes.append(p.size())
                g = g.view(-1) # flatten vector
                p = p.view(-1) # flatten vector
                assert( len(g.size())==1 and len(p.size())==1 )
                assert( g.size(0)== p.size(0) )
                flatten_lengths.append( p.size(0) )
                flatten_grads.append(g)
                flatten_params.append(p)
        flatten_params = torch.cat(flatten_params, dim=0)
        flatten_grads = torch.cat(flatten_grads, dim=0)
        n_learner_params = flatten_params.size(0)
        # hx i.e. previous forget, update & cell state from metalstm
        if None in metalstm_hx:
            # set initial f_prev, i_prev, c_prev]
            metalstm_hx = trainable_opt_model.initialize_meta_lstm_cell(flatten_params)
        # preprocess grads
        grad_prep = preprocess_grad_loss(flatten_grads)  # [|theta_p|, 2]
        loss_prep = preprocess_grad_loss(inner_train_loss) # [1, 2]
        err_prep = preprocess_grad_loss(inner_train_err) # [1, 2]
        # get new parameters from meta-learner/trainable optimizer/meta-lstm optimizer
        theta_next, [(lstmh, lstmc), metalstm_hsx] = trainable_opt_model(
            inputs=[loss_prep, grad_prep, err_prep, flatten_grads],
            hs=[(lstmh, lstmc), metalstm_hx]
        )
        #assert( theta_next )
        # start differentiable & trainable update
        zipped = zip(self.param_groups, grouped_grads)
        i = 0
        for group_idx, (group, grads) in enumerate(zipped):
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue
                ## update params with new theta: f^<t>*theta^<t-1> - i^<t>*grad^<t>
                original_size = original_sizes[p_idx]
                p_len = flatten_lengths[p_idx]
                assert(p_len == np.prod(original_size))
                p_new  = theta_next[i:i+p_len]
                p_new = p_new.view(original_size)
                group['params'][p_idx] = p_new
                i = i+p_len
        assert(i == n_learner_params)
        # fake returns
        self.param_groups[0]['kwargs']['trainable_opt_state']['prev_lstm_state'] = [(lstmh, lstmc), metalstm_hx]

class MetaLSTMCell(nn.Module):
    """
    Based on: https://github.com/brando90/meta-learning-lstm-pytorch
    Or: https://github.com/markdtw/meta-learning-lstm-pytorch

    Model:
    C_t = f_t * C_{t-1} + i_t * \tilde{C_t}
    """
    def __init__(self, device, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size # equal to first layer lstm.hidden_size e.g. 32
        self.hidden_size = hidden_size
        assert(self.hidden_size == 1)
        self.WF = nn.Parameter(torch.Tensor(input_size + 2, hidden_size)) # [input_size+2, 1]
        self.WI = nn.Parameter(torch.Tensor(input_size + 2, hidden_size)) # [input_size+2, 1]
        #self.cI = nn.Parameter(torch_uu.Tensor(n_learner_params, 1)) # not needed because training init is handled somewhere ese
        self.bI = nn.Parameter(torch.Tensor(1, hidden_size))
        self.bF = nn.Parameter(torch.Tensor(1, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        ## reset all the params of meta-lstm trainer (including cI init of the base/child net if they are included in the constructor)
        for weight in self.parameters():
            nn.init.uniform_(weight, -0.01, 0.01)

        ## want initial forget value to be high and input value to be low so that model starts with gradient descent
        ## f^<t>*c^<t-1> + i^<t>*c~^<t> =~= theta^<t-1> - i^<t> * grad^<t>
        # set unitial forget value to be high so that f^<t>*theta^<t> = 1*theta^<t>
        nn.init.uniform_(self.bF, a=4, b=6) # samples from Uniform(a,b)
        # set initial learning rate to be low so its approx GD with small step size at start
        nn.init.uniform_(self.bI, -5, -4) # samples from Uniform(a,b)

    def init_cI(self, flat_params):
        self.cI.data.copy_(flat_params.unsqueeze(1))

    def forward(self, inputs, hx=None):
        # gunpack inputs
        lstmh, grad = inputs # (lstm(grad_t, loss_t), grad)
        n_learner_params, input_size = lstmh.size(1), lstmh.size(2) # note input_size = hidden_size for first layer lstm. Hidden size for meta-lstm learner is 1.
        grad = grad.view(n_learner_params, 1) # dim -> [n_learner_params, 1]
        f_prev, i_prev, c_prev = hx
        # change dim so matrix mult mm work: dim -> [n_learner_params, 1]
        lstmh = lstmh.view(n_learner_params, input_size) # -> [input_size x n_learner_params] = [hidden_size x n_learner_params]
        f_prev, i_prev, c_prev = f_prev.view(n_learner_params,1), i_prev.view(n_learner_params,1), c_prev.view(n_learner_params,1)
        # f_t = sigmoid(W_f * [ lstm(grad_t, loss_t), theta_{t-1}, f_{t-1}] + b_f)
        f_next = sigmoid( cat((lstmh, c_prev, f_prev), 1).mm(self.WF) + self.bF.expand_as(f_prev) ) # [n_learner_params, 1] = [n_learner_params, input_size+3] x [input_size+3, 1] + [n_learner_params, 1]
        # i_t = sigmoid(W_i * [ lstm(grad_t, loss_t), theta_{t-1}, i_{t-1}] + b_i)
        i_next = sigmoid( cat((lstmh, c_prev, i_prev), 1).mm(self.WI) + self.bI.expand_as(i_prev) ) # [n_learner_params, 1] = [n_learner_params, input_size+3] x [input_size+3, 1] + [n_learner_params, 1]
        # next cell/params: theta^<t> = f^<t>*theta^<t-1> - i^<t>*grad^<t>
        c_next = f_next*(c_prev) - i_next*(grad) # note, - sign is important cuz i_next is positive due to sigmoid activation

        # c_next.squeeze() left for legacydsgdfagsdhsjsjhdfhjdhgjfghjdgj
        #c_next = c_next.squeeze()
        assert(c_next.size() == torch.Size((n_learner_params,1)))
        return c_next.squeeze(), [f_next, i_next, c_next]

    def extra_repr(self):
        s = '{input_size}, {hidden_size}, {n_learner_params}'
        return s.format(**self.__dict__)

class MetaLstmOptimizer(nn.Module):
    '''
    Meta-learner/optimizer based on Optimization as a model for few shot learning.
    '''

    def __init__(self, device, input_size, hidden_size, num_layers=1):
        """Args:
            input_size (int): for the first LSTM layer, 6
            hidden_size (int): for the first LSTM layer, e.g. 32
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers
        ).to(device)
        #self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size).to(device)
        self.metalstm = MetaLSTMCell(device=device, input_size=hidden_size, hidden_size=1).to(device)
        self.to(device)

    def forward(self, inputs, hs=None):
        loss_prep, grad_prep, err_prep, grad = inputs
        ## forward pass of first lstm
        # sort out input x^<t> to normal lstm
        loss_prep = loss_prep.expand_as(grad_prep) # [1, 2] -> [n_learner_params, 2]
        err_prep = err_prep.expand_as(grad_prep) # [1, 2] -> [n_learner_params, 2]
        xn_lstm = torch.cat((loss_prep, err_prep, grad_prep), 1).unsqueeze(0) # [n_learner_params, 6]
        # normal lstm([loss, grad_prep, train_err]) = lstm(xn)
        n_learner_params = xn_lstm.size(1)
        (lstmh, lstmc) = hs[0] # previous hx from first (standard) lstm i.e. lstm_hx = (lstmh, lstmc) = hs[0]
        if lstmh.size(1) != xn_lstm.size(1): # only true when prev lstm_hx is equal to decoder/controllers hx
            # make sure that h, c from decoder/controller has the right size to go into the meta-optimizer
            expand_size = torch.Size([1,n_learner_params,self.lstm.hidden_size])
            lstmh, lstmc = lstmh.squeeze(0).expand(expand_size).contiguous(), lstmc.squeeze(0).expand(expand_size).contiguous()
        lstm_out, (lstmh, lstmc) = self.lstm(input=xn_lstm, hx=(lstmh, lstmc))

        ## forward pass of meta-lstm i.e. theta^<t> = f^<t>*theta^<t-1> + i^<t>*grad^<t>
        metalstm_hx = hs[1] # previous hx from optimizer/meta lstm = [metalstm_fn, metalstm_in, metalstm_cn]
        xn_metalstm = [lstmh, grad] # note, the losses,grads are preprocessed by the lstm first before passing to metalstm [outputs_of_lstm, grad] = [ lstm(losses, grad_preps), grad]
        theta_next, metalstm_hx = self.metalstm(inputs=xn_metalstm, hx=metalstm_hx)

        return theta_next, [(lstmh, lstmc), metalstm_hx]

    def get_trainable_opt_state(self, out, h, c, *args, **kwargs):
        inner_opt, args = kwargs['inner_opt'], kwargs['args']
        # process hidden state from arch decoder/controller
        h = ( ( out.mean()+h )/2 ).expand_as(h)
        # if lstmh.size() != xn_lstm.size():
        #     lstmh = torch_uu.cat((lstmh,lstmh,lstmh),dim=2).squeeze(0).expand_as(xn_lstm)
        #     lstmc = torch_uu.cat((lstmc,lstmc,lstmc),dim=2).squeeze(0).expand_as(xn_lstm)
        trainable_opt_state = {}
        # reset h, c to be the decoders out & h
        [(lstmh, lstmc), metalstm_hx] = [(h, c), (None, None, None)]
        if inner_opt is None: # inner optimizer has not been used yet
            pass
        else: # use info from a the inner optimizer from previous outer loop
            # if you want to use the prev inner optimizer's & decoder/controler's (h, c)
            #h, c = (h + h_opt.detach())/2, (c + c_opt.detach())/2
            pass
        trainable_opt_state['prev_state'] = [(lstmh, lstmc), metalstm_hx]
        ## create initial trainable opt state
        return trainable_opt_state

    def initialize_meta_lstm_cell(self, flatten_params):
        device = flatten_params.device
        n_learner_params = flatten_params.size(0)
        f_prev = torch.zeros((n_learner_params, self.metalstm.hidden_size)).to(device)
        i_prev = torch.zeros((n_learner_params, self.metalstm.hidden_size)).to(device)
        c_prev = flatten_params
        meta_hx = [f_prev, i_prev, c_prev]
        return meta_hx

higher.register_optim(EmptyMetaLstmOptimizer, MetaTrainableLstmOptimizer)

####
####

class EmptyMAML(Optimizer):

    def __init__(self, params, trainable_opt_model, trainable_opt_state, *args, **kwargs):
        defaults = {
            'trainable_opt_model':trainable_opt_model, 
            'trainable_opt_state':trainable_opt_state, 
            'args':args, 
            'kwargs':kwargs
        }
        super().__init__(params, defaults)

class MAMLMetaOptimizer(DifferentiableOptimizer):

    def _update(self, grouped_grads, **kwargs):
        #prev_lr = self.param_groups[0]['trainable_opt_state']['prev_lr']
        maml_opt_mdl = self.param_groups[0]['trainable_opt_model'] # basically just the learning rate
        # start differentiable & trainable update
        zipped = zip(self.param_groups, grouped_grads)
        inner_lr = maml_opt_mdl() # simply return the inner step size/learning rate lr
        for group_idx, (group, grads) in enumerate(zipped):
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue
                p_new = p - inner_lr*g
                group['params'][p_idx] = p_new
        # fake returns
        #self.param_groups[0]['trainable_opt_state']['prev_lr'] = inner_lr

class MAMLOptModel(nn.Module):

    def __init__(self, device, initial_inner_lr=0.1, requires_grad=False):
        """Our trainable implementation of MAML

        Arguments:
            device {[type]} -- device

        Keyword Arguments:
            initial_eta {float} -- initial step size (default: {0.1}), default it's MAMLs for mini-imagenet. Classifier baseline paper suggests it to be large 0.1
            requires_grad {bool} -- true if you want to train the learning rate number (default: {False}) (note there is no sigmoid to avoid it becoming negative)
        """
        super().__init__()
        initial_inner_lr = torch.tensor([initial_inner_lr], device=device, requires_grad=requires_grad)
        self.inner_lr = nn.Parameter( initial_inner_lr, requires_grad=requires_grad)
        self.inner_lr = self.inner_lr.to(device)
        self.device = device
        if len(self.inner_lr.size()) >= 2:
            raise ValueError(f'Learnign rate for MAML has to be a 1D number but was of shape {self.eta.size()}')

    def forward(self):
        inner_lr = self.inner_lr
        return inner_lr

    def get_trainable_opt_state(self, out, h, c, *args, **kwargs):
        #prev_lr = ((out.mean()+h.mean()+c.mean())/3).view(1)
        #trainable_opt_state = {'prev_lr': None}
        trainable_opt_state = {}
        return trainable_opt_state

higher.register_optim(EmptyMAML, MAMLMetaOptimizer)

class EmptySimpleMetaLstm(Optimizer):

    def __init__(self, params, *args, **kwargs):
        defaults = { 'args':args, 'kwargs':kwargs}
        super().__init__(params, defaults)

class SimpleMetaLstm(DifferentiableOptimizer):

    def _update(self, grouped_grads, **kwargs):
        prev_lr = self.override['trainable_opt_state']['prev_lr']
        simp_meta_lstm = self.override['trainable_opt_model']
        # start differentiable & trainable update
        zipped = zip(self.param_groups, grouped_grads)
        for group_idx, (group, grads) in enumerate(zipped):
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue
                # get gradient as "data"
                g = g.detach() # gradients of gradients are not used (no hessians)
                ## very simplified version of meta-lstm meta-learner
                input_metalstm = torch.stack([p, g, prev_lr.view(1,1)]).view(1,3) # [p, g, prev_lr] note it's missing loss, normalization etc. see original paper
                lr = simp_meta_lstm(input_metalstm).view(1)
                fg = 1 - lr # learnable forget rate
                ## update suggested by meta-lstm meta-learner
                p_new = fg*p - lr*g
                group['params'][p_idx] = p_new
        # fake returns
        self.override['trainable_opt_state']['prev_lr'] = lr

higher.register_optim(EmptySimpleMetaLstm, SimpleMetaLstm)

####
####

def test_parametrized_inner_optimizer():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from collections import OrderedDict

    ## training config
    #device = torch_uu.device("cuda" if torch_uu.cuda.is_available() else "cpu")
    track_higher_grads = True # if True, during unrolled optimization the graph be retained, and the fast weights will bear grad funcs, so as to permit backpropagation through the optimization process. False during test time for efficiency reasons
    copy_initial_weights = False # if False then we train the base models initial weights (i.e. the base model's initialization)
    episodes = 5
    nb_inner_train_steps = 5
    ## get base model
    base_mdl = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1,1, bias=False)),
        ('act', nn.LeakyReLU()),
        ('fc2', nn.Linear(1,1, bias=False))
        ]))
    ## parametrization/mdl for the inner optimizer
    opt_mdl = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(3,1, bias=False)), # 3 inputs [p, g, prev_lr] 1 for parameter, 1 for gradient, 1 for previous lr
        ('act', nn.LeakyReLU()),
        ('fc2', nn.Linear(1,1, bias=False))
        ]))
    ## get outer optimizer (not differentiable nor trainable)
    outer_opt = optim.Adam([{'params': base_mdl.parameters()},{'params': opt_mdl.parameters()}], lr=0.01)
    for episode in range(episodes):
        ## get fake support & query data (from a single task and 1 data point)
        spt_x, spt_y, qry_x, qry_y = torch.randn(1), torch.randn(1), torch.randn(1), torch.randn(1)
        ## get differentiable & trainable (parametrized) inner optimizer
        inner_opt = EmptySimpleMetaLstm( base_mdl.parameters() )
        with higher.innerloop_ctx(base_mdl, inner_opt, copy_initial_weights=copy_initial_weights, track_higher_grads=track_higher_grads) as (fmodel, diffopt):
            diffopt.override = {'trainable_opt_model': opt_mdl, 'trainable_opt_state': {'prev_lr': 0.9*torch.randn(1)} }
            for i_inner in range(nb_inner_train_steps): # this current version implements full gradient descent on k_shot examples (which is usually small  5)
                fmodel.train()
                # base/child model forward pass
                inner_loss = 0.5*((fmodel(spt_x) - spt_y))**2
                # inner-opt update
                diffopt.step(inner_loss)
            ## Evaluate on query set for current task
            qry_loss = 0.5*((fmodel(qry_x) - qry_y))**2
            qry_loss.backward() # for memory efficient computation
        ## outer update
        print(f'episode = {episode}')
        print(f'ase_mdl.fc1.weight.grad = {base_mdl.fc1.weight.grad}')
        print(f'ase_mdl.fc2.weight.grad = {base_mdl.fc2.weight.grad}')
        print(f'opt_mdl.fc1.weight.grad = {opt_mdl.fc1.weight.grad}')
        print(f'opt_mdl.fc2.weight.grad = {opt_mdl.fc2.weight.grad}')
        outer_opt.step()
        outer_opt.zero_grad()

def test_training_initial_weights():
    import torch
    import torch.optim as optim
    import torch.nn as nn
    from collections import OrderedDict

    ## training config
    #device = torch_uu.device("cuda" if torch_uu.cuda.is_available() else "cpu")
    episodes = 5
    nb_inner_train_steps = 5
    ## get base model
    base_mdl = nn.Sequential(OrderedDict([
        ('fc', nn.Linear(1,1, bias=False)),
        ('relu', nn.ReLU())
        ]))
    ## get outer optimizer (not differentiable nor trainable)
    outer_opt = optim.Adam(base_mdl.parameters(), lr=0.01)
    for episode in range(episodes):
        spt_x, spt_y, qry_x, qry_y = torch.randn(1), torch.randn(1), torch.randn(1), torch.randn(1)
        inner_opt = torch.optim.SGD(base_mdl.parameters(), lr=1e-1)
        with higher.innerloop_ctx(base_mdl, inner_opt, copy_initial_weights=False, track_higher_grads=False) as (fmodel, diffopt):
            for i_inner in range(nb_inner_train_steps): # this current version implements full gradient descent on k_shot examples (which is usually small  5)
                fmodel.train()
                # base/child model forward pass
                inner_loss = 0.5*((fmodel(spt_x) - spt_y))**2
                # inner-opt update
                diffopt.step(inner_loss)
            ## Evaluate on query set for current task
            qry_loss = 0.5*((fmodel(qry_x) - qry_y))**2
            qry_loss.backward() # for memory efficient computation
        ## outer update
        print(f'episode = {episode}')
        print(f'base_mdl.grad = {base_mdl.fc.weight.grad}')
        outer_opt.step()
        outer_opt.zero_grad()

def so_example_about_copy_initial_weights():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import higher
    import numpy as np

    np.random.seed(1)
    torch.manual_seed(3)
    N = 100
    actual_multiplier = 3.5
    meta_lr = 0.00001
    loops = 5 # how many iterations in the inner loop we want to do

    x = torch.tensor(np.random.random((N,1)), dtype=torch.float64) # features for inner training loop
    y = x * actual_multiplier # target for inner training loop
    model = nn.Linear(1, 1, bias=False).double() # simplest possible model - multiple input x by weight w without bias
    meta_opt = optim.SGD(model.parameters(), lr=meta_lr, momentum=0.)

    def run_inner_loop_once(model, verbose, copy_initial_weights):
        lr_tensor = torch.tensor([0.3], requires_grad=True)
        momentum_tensor = torch.tensor([0.5], requires_grad=True)
        opt = optim.SGD(model.parameters(), lr=0.3, momentum=0.5)
        with higher.innerloop_ctx(model, opt, copy_initial_weights=copy_initial_weights, override={'lr': lr_tensor, 'momentum': momentum_tensor}) as (fmodel, diffopt):
            for j in range(loops):
                if verbose:
                    print('Starting inner loop step j=={0}'.format(j))
                    print('    Representation of fmodel.parameters(time={0}): {1}'.format(j, str(list(fmodel.parameters(time=j)))))
                    print('    Notice that fmodel.parameters() is same as fmodel.parameters(time={0}): {1}'.format(j, (list(fmodel.parameters())[0] is list(fmodel.parameters(time=j))[0])))
                out = fmodel(x)
                if verbose:
                    print('    Notice how `out` is `x` multiplied by the latest version of weight: {0:.4} * {1:.4} == {2:.4}'.format(x[0,0].item(), list(fmodel.parameters())[0].item(), out[0].item()))
                loss = ((out - y)**2).mean()
                diffopt.step(loss)

            if verbose:
                # after all inner training let's see all steps' parameter tensors
                print()
                print("Let's print all intermediate parameters versions after inner loop is done:")
                for j in range(loops+1):
                    print('    For j=={0} parameter is: {1}'.format(j, str(list(fmodel.parameters(time=j)))))
                print()

            # let's imagine now that our meta-learning optimization is trying to check how far we got in the end from the actual_multiplier
            weight_learned_after_full_inner_loop = list(fmodel.parameters())[0]
            meta_loss = (weight_learned_after_full_inner_loop - actual_multiplier)**2
            print('  Final meta-loss: {0}'.format(meta_loss.item()))
            meta_loss.backward() # will only propagate gradient to original model parameter's `grad` if copy_initial_weight=False
            if verbose:
                print('  Gradient of final loss we got for lr and momentum: {0} and {1}'.format(lr_tensor.grad, momentum_tensor.grad))
                print('  If you change number of iterations "loops" to much larger number final loss will be stable and the values above will be smaller')
            return meta_loss.item()

    print('=================== Run Inner Loop First Time (copy_initial_weights=True) =================\n')
    meta_loss_val1 = run_inner_loop_once(model, verbose=True, copy_initial_weights=True)
    print("\n---> Let's see if we got any gradient for initial model parameters: {0}\n".format(list(model.parameters())[0].grad))
    print(f'--> = {model.weight.grad}')

    print('=================== Run Inner Loop Second Time (copy_initial_weights=False) =================\n')
    meta_loss_val2 = run_inner_loop_once(model, verbose=False, copy_initial_weights=False)
    print("\n---> Let's see if we got any gradient for initial model parameters: {0}\n".format(list(model.parameters())[0].grad))
    print(f'--> = {model.weight.grad}')

    print('=================== Run Inner Loop Third Time (copy_initial_weights=False) =================\n')
    final_meta_gradient = list(model.parameters())[0].grad.item()
    # Now let's double-check `higher` library is actually doing what it promised to do, not just giving us
    # a bunch of hand-wavy statements and difficult to read code.
    # We will do a simple SGD step using meta_opt changing initial weight for the training and see how meta loss changed
    meta_opt.step()
    meta_opt.zero_grad()
    meta_step = - meta_lr * final_meta_gradient # how much meta_opt actually shifted inital weight value
    meta_loss_val3 = run_inner_loop_once(model, verbose=False, copy_initial_weights=False)

    meta_loss_gradient_approximation = (meta_loss_val3 - meta_loss_val2) / meta_step

    print()
    print('Side-by-side meta_loss_gradient_approximation and gradient computed by `higher` lib: {0:.4} VS {1:.4}'.format(meta_loss_gradient_approximation, final_meta_gradient))

def testing_higher_order_grads():
    import torch 
    x = torch.tensor([4.], requires_grad=True) 
    x_cubed = x * x * x 
    first_grad = torch.autograd.grad(x_cubed, x, retain_graph=True, create_graph=True) 
    second_grad = torch.autograd.grad(first_grad, x, retain_graph=True, create_graph=True) 
    third_grad = torch.autograd.grad(second_grad, x, retain_graph=True, create_graph=True) 

    print(first_grad) 
    print(second_grad) 
    print(third_grad)

if __name__ == '__main__':
    test_parametrized_inner_optimizer()
    #test_training_initial_weights()
    #so_example_about_copy_initial_weights()
    testing_higher_order_grads()
    print('Done \a')