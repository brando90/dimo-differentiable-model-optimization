"""
This is a module for performance predictors that are ONLY evaluators (i.e. do not learn)
"""

import torch.nn as nn

class EvaluatorFunction(nn.Module):
    """ Class that is an object version of a lambda function for calling the loss. e.g.
    e.g.
        criterion = nn.CrossEntropyLoss()
        performance_predictor = lambda outputs, labels: criterion(outputs, labels) # val_loss

    replaced by:
        criterion = nn.CrossEntropyLoss()
        performance_predictor = EvaluatorFunction(criterion)


    Reason is because dill needs to know how to fill in the lambda function body (that might reference values outside of it) 
    when it unpickles because it doesn't save the program state

    Read further details: https://stackoverflow.com/questions/61510810/how-does-one-pickle-arbitrary-pytorch-models-that-use-lambda-functions/61523763#61523763
    """

    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, outputs, labels):
        loss = self.criterion(outputs, labels)
        return loss
