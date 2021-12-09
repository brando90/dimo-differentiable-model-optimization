#__all__ = ["DiMO"]

import torch.nn as nn

#from torchviz import make_dot

import higher

#from torch_uu import lp_norm

from uutils.torch_uu import calc_accuracy, calc_error


class MetaLearner(nn.Module):
#lass MetaLearner:
    """
    Meta-Learner according to DiMO.
    """
    def __init__(
        self,
        args,

        x_thought, 
        
        controller, 
        sampler, 

        child_model,
        
        inner_opt_constructor, 
        add_inner_train_info, 
        opt_model, 
        
        performance_predictor
    ):
        """WARNING: if you are picling lambda functions make sure they are pure functions or you talk to Brando if they are not.

        Reference: https://stackoverflow.com/questions/61510810/how-does-one-pickle-arbitrary-pytorch-models-that-use-lambda-functions/61523763#61523763
        """
        super().__init__()
        self.args = args # args for experiment
        if x_thought is not None:
            x_thought = nn.Parameter(x_thought)
        if not isinstance(x_thought, nn.Parameter):
            x_thought = nn.Parameter(x_thought)
        self.x_thought = x_thought # might have nn.Parameter's
        
        self.controller = controller # might have nn.Parameter's
        self.sampler = sampler
        self.child_model = child_model
        
        self.inner_opt_constructor = inner_opt_constructor # WARNING: if these are functions make sure they are pure functions
        self.add_inner_train_info = add_inner_train_info # WARNING: if these are functions make sure they are pure functions
        self.opt_model = opt_model # might have nn.Parameter's
        
        # WARNING: if this is a function make sure its are pure functions or a class with only pure functions
        self.performance_predictor = performance_predictor # might have nn.Parameter's, 

        self.inner_debug = False

    def forward(self, inner_inputs, inner_targets, outer_inputs, outer_targets):
        """

        Forward pass = Inner Train Child Mdl -> Performance Predictor -> Meta-Loss
        
        Arguments:
            child_model {[type]} -- [description]
            inner_inputs {[type]} -- [description]
            inner_targets {[type]} -- [description]
        """
        ## Get Get Inner Optimizer
        inner_opt = self.get_inner_opt(self.child_model, controller_hx=self.controller_hx)
        ## Inner Train
        with higher.innerloop_ctx(self.child_model, inner_opt, copy_initial_weights=self.args.copy_initial_weights, track_higher_grads=self.args.track_higher_grads) as (fmodel, diffopt):
            for inner_epoch in range(self.args.nb_inner_train_epochs):
                self.args.inner_i = 0
                for batch_idx in range(0, len(inner_inputs), self.args.batch_size):
                    #print(f'inner_i, batch_idx = {inner_i, batch_idx}')
                    fmodel.train()
                    # get batch for inner training
                    inner_input = inner_inputs[batch_idx:batch_idx+self.args.batch_size].to(self.args.device)
                    inner_target = inner_targets[batch_idx:batch_idx+self.args.batch_size].to(self.args.device)
                    # child model forward pass
                    logits = fmodel(inner_input)
                    inner_loss = self.args.criterion(logits, inner_target)
                    inner_train_err = calc_error(mdl=fmodel, X=outer_inputs, Y=outer_targets)
                    # inner-opt update
                    self.add_inner_train_info(diffopt, inner_train_loss=inner_loss, inner_train_err=inner_train_err)
                    diffopt.step(inner_loss)
                    if self.inner_debug:
                        inner_train_acc = calc_accuracy(mdl=fmodel, X=inner_input, Y=inner_target)
                        self.args.logger.loginfo(f'Inner:[inner_i={self.args.inner_i}], inner_loss: {inner_loss}, inner_train_acc: {inner_train_acc}, test loss: {-1}, test acc: {-1}')
                    self.args.inner_i += 1
                ## Evaluate on test dataset/performance predictor
                fmodel.train() if self.args.mode == 'meta-train' else fmodel.eval()
                outputs = fmodel(outer_inputs)
                meta_loss = self.performance_predictor(outputs, outer_targets)
                outer_train_acc = calc_accuracy(mdl=fmodel, X=outer_inputs, Y=outer_targets)
        return self.child_model, meta_loss, outer_train_acc

    def get_child_model(self):
        if self.args.child_model_mode == 'fresh_chosen_child_model_from_controller_each_outer_loop':
            # generate the arch: i.e. the conv cells and reduc cells
            conv, reduc, out, h, c = self.controller(x_thought=self.x_thought)
            # sample totally new/fresh child model from arch/conv-reduc
            self.child_model = self.sampler(conv, reduc, self.args.num_nodes, self.args.dropout)
            # IMPORTANT: we are not saving this arch because for changing arch we only save the one that performs best on validation set. See log_validation code in training_loop
        elif self.args.child_model_mode == "training_init_for_child_model":
            # generate the arch: i.e. the conv cells and reduc cells
            conv, reduc, out, h, c = self.controller(x_thought=self.x_thought)
            # sample a new child model according to arch/conv-reduc but share weights/initialization
            self.child_model.update_network(conv, reduc)
            # IMPORTANT: we are not saving this arch because for changing arch we only save the one that performs best on validation set. See log_validation code in training_loop
        else:
            ## Use fix child_model
            # with a fixed child model we don't need to connect the meta-learner forward pass with the controller, etc.
            out, h, c = self.args.out.detach(), self.args.h.detach(), self.args.c.detach() # not usually needed because the decoder creates new h, c **from strach at the start of every outer loop**, this is needed because otherwise the computations graphs btw outer loops are accidentally connected (gives .backward error)
            self.child_model = self.args.child_model
            # IMPORTANT: the arch was saved already at the beginning of running this script. No need to save it again.
        self.controller_hx = (out, h, c)

    def get_inner_opt(self, child_model, **kwargs):
        """
        The inner opt works as follow:
        1) it receives (implicitly through torch_uu globals that we modify) the next state (though the first one we need to initialize it with self.opt_model.get_trainable_opt_state)
        2) pass that and the child mode params to the inner opt

        Then it's ready to be passed to higher to be made differentiable.

        Returns:
            [Optimizer] -- trainable inner optimizer
        """
        ## Get initial state of inner optimizer
        (out, h, c) = kwargs['controller_hx']
        inner_opt = None # <- DEFAULT: uses controller's (out,hx) to initialize the state of inner optimizer
        #inner_opt = self.inner_opt # uses controller's (out,hx) AND the previous diff/inner opt (out,hx) to initialize the state of inner optimizer 
        initial_opt_state = self.opt_model.get_trainable_opt_state(out=out, h=h, c=c, inner_opt=inner_opt, child_model=child_model, args=self.args)
        ## Construct inner opt with the parametetrs its (differentiably) updating
        child_model_params = [{'params':child_model.parameters()}]
        self.inner_opt = self.inner_opt_constructor(child_model_params, trainable_opt_model=self.opt_model, trainable_opt_state=initial_opt_state)
        return self.inner_opt

    def parameters(self):
        # TODO, check that it does get the meta-param correctly and then uncomment this
        raise ValueError('NOT CHECK IF THIS CLASS GETS ALL DIMO PARAM PROPERLY!')

def test_parameters_gottten_correctly():
    pass # TODO

if __name__ == "__main__":
    test_parameters_gottten_correctly()