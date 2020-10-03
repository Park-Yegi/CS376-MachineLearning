import torch


class NN(torch.nn.Module):
    """
    Your Neural Netowrk model.
    """
    def __init__(self):
        super(NN, self).__init__()
        raise NotImplementedError

    def forward(self, X):
        """
        X: Data. A pytorch Tensor of dimensions [number of samples, number of features].
        :return: Output of your neural network model. A pytorch Tensor of dimensions [number of samples].
        
        """
        raise NotImplementedError


def train(model, X, y):
    """
    model: Neural network model. An instance of the NN class.
    X: Data. A pytorch Tensor of dimensions [number of samples, number of features].
    y: Targets. A pytorch Tensor of dimensions [number of samples].
    :return: Nothing.
    """
    raise NotImplementedError


def test(model, X):
    """
    model: Neural network model. An instance of the NN class.
    X: Data. A pytorch Tensor of dimensions [number of samples, number of features].
    :return: Predicted targets. A pytorch Tensor of dimensions [number of samples].
    """
    raise NotImplementedError