#__all__ = ["DiMO"]

import torch.nn as nn

#from torchviz import make_dot

import higher

#from torch_uu import lp_norm

from uutils.torch_uu import calc_accuracy, calc_error


class MetaLearner(nn.Module):
    """
    Meta-Learner according to DiMO.
    """
    def __init__(
        self,
        args,

        base_model,
        
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

        self.base_model = base_model
        
        self.inner_opt_constructor = inner_opt_constructor # WARNING: if these are functions make sure they are pure functions
        self.add_inner_train_info = add_inner_train_info # WARNING: if these are functions make sure they are pure functions
        self.opt_model = opt_model # might have nn.Parameter's
        
        # WARNING: if this is a function make sure its are pure functions or a class with only pure functions
        self.performance_predictor = performance_predictor # might have nn.Parameter's, 

        self.inner_debug = False

        # note that this param is usually the controller_hx when it samples an base_model from the NAS controller
        # it cal also be the last hx from the previous outer loop step at the final inner loop i.e. opt^<t_o-1,T>.hx
        # or it can be fixed zero or random
        self.initial_inner_opt_hx = (None, None, None)

    def forward(self, S_x, S_y, Q_x, Q_y):
        """
        Forward pass = Inner Train Child Mdl -> Performance Predictor -> Meta-Loss
        
        Arguments:
            S_x {tensor([k_shot*N,C,H,W])} - Support set with k_shot image examples for each of the N classes. |S_x| = k_shot*N e.g. torch_uu.Size([25, 3, 84, 84])
            S_y {tensor([k_shot*N])} - Support set with k_shot target examples for each of the N classes. |S_y| = k_shot*N e.g. torch_uu.Size([25])
        """
        ## Get Get Inner Optimizer
        inner_opt = self.get_inner_opt(self.base_model, initial_inner_opt_hx=self.initial_inner_opt_hx)
        ## Inner Train
        with higher.innerloop_ctx(self.base_model, inner_opt, copy_initial_weights=self.args.copy_initial_weights, track_higher_grads=self.args.track_higher_grads) as (fmodel, diffopt):
            for inner_epoch in range(self.args.nb_inner_train_steps):
                self.args.inner_i = 0
                for batch_idx in range(0, len(S_x), self.args.batch_size):
                    #print(f'inner_i, batch_idx = {inner_i, batch_idx}')
                    fmodel.train()
                    # get batch for inner training, usually with support/innner set
                    inner_input = S_x[batch_idx:batch_idx+self.args.batch_size].to(self.args.device)
                    inner_target = S_y[batch_idx:batch_idx+self.args.batch_size].to(self.args.device)
                    # base/child model forward pass
                    logits = fmodel(inner_input)
                    inner_loss = self.args.criterion(logits, inner_target)
                    inner_train_err = calc_error(mdl=fmodel, X=S_x, Y=S_y) # careful, use Q_x during meta-train and Q_x during meta-eval
                    # inner-opt update
                    self.add_inner_train_info(diffopt, inner_train_loss=inner_loss, inner_train_err=inner_train_err)
                    diffopt.step(inner_loss)
                    if self.inner_debug:
                        inner_train_acc = calc_accuracy(mdl=fmodel, X=inner_input, Y=inner_target)
                        self.args.logger.loginfo(f'Inner:[inner_i={self.args.inner_i}], inner_loss: {inner_loss}, inner_train_acc: {inner_train_acc}, test loss: {-1}, test acc: {-1}')
                    self.args.inner_i += 1
                ## Evaluate on test dataset/performance predictor
                fmodel.train() if self.args.mode == 'meta-train' else fmodel.eval()
                outputs = fmodel(Q_x)
                meta_loss = self.performance_predictor(outputs, Q_y)
                outer_train_acc = calc_accuracy(mdl=fmodel, X=Q_x, Y=Q_y)
        return self.base_model, meta_loss, outer_train_acc

    def get_inner_opt(self, base_model, *args, **kwargs):
        """
        The inner opt works as follow:
        1) it receives (implicitly through torch_uu globals that we modify) the next state (though the first one we need to initialize it with self.opt_model.get_trainable_opt_state)
        2) pass that and the child mode params to the inner opt

        Then it's ready to be passed to higher to be made differentiable.

        Returns:
            [Optimizer] -- trainable inner optimizer
        """
        ## Get initial state hx of inner optimizer
        (out, h, c) = kwargs['initial_inner_opt_hx'] # when doing DiMO usually it's controller_hx
        inner_opt = None # <- DEFAULT: uses controller's (out,hx) to initialize the state of inner optimizer
        #inner_opt = self.inner_opt # uses controller's (out,hx) AND the previous diff/inner opt (out,hx) to initialize the state of inner optimizer 
        initial_opt_state = self.opt_model.get_trainable_opt_state(out=out, h=h, c=c, inner_opt=inner_opt, base_model=base_model, args=self.args)
        ## Construct inner opt with the parametetrs its (differentiably) updating
        base_model_params = [{'params':base_model.parameters()}]
        self.inner_opt = self.inner_opt_constructor(base_model_params, trainable_opt_model=self.opt_model, trainable_opt_state=initial_opt_state)
        return self.inner_opt

def test():
    print("TODO")

if __name__ == "__main__":
    test()
