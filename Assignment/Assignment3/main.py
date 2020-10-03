# from elice_utils import EliceUtils

# elice_utils = EliceUtils()

import numpy as np
# import torch
# import clustering
import mlp_numpy
# import nn_pytorch


def main():
    """
    The purpose of main.py is to provide an running example of classes and functions.
    This file is not considered in the evaluation process.
    """

    # data generated from 3 mixture components
    # dim : n_samples X n_features
    X = np.random.normal([[[1], [1]], [[2], [2]], [[3], [3]]], 0.2, (3, 2, 100)).swapaxes(1, 2).reshape(300, 2) # 300 samples, 2 features
    X = np.random.normal([[[1], [1], [1]], [[2], [2], [2]], [[3], [3], [3]]], 0.2, (3, 3, 1000)).swapaxes(1, 2).reshape(3000, 3) # 3000 samples, 3 features

    # K_means
    # model = clustering.K_means()
    # model.fit(X, 3)
    # centers = model.get_centers()

    # EM
#     model = clustering.EM()
#     model.fit(X, 4)
#     pis, mus, sigmas = model.get_params()
    
#     print("Main PIS")
#     print(pis)
#     print("Main MUS")
#     print(mus)
#     print("Main SIGMAS")
#     print(sigmas)

    # get samples from data/nn_train.txt and data/nn_test.txt
    X_data = np.loadtxt("./data/nn_train.txt")
    X_train = np.delete(X_data, 0, 1)
    y_train = np.delete(X_data, [1, 2], 1)
    X_test = np.loadtxt("./data/nn_test.txt")
    # X_data = torch.tensor(np.loadtxt("./data/nn_train.txt"))
    # X_train = torch.tensor(np.delete(X_data, 0, 1)).float()
    # y_train = torch.tensor(np.delete(X_data, [1, 2], 1)).float()
    # X_test = torch.tensor(np.loadtxt("./data/nn_test.txt")).float()
    
    # MLP
    model = mlp_numpy.MLP()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # NN
    # model = nn_pytorch.NN()
    # nn_pytorch.train(model, X_train, y_train)
    # y_pred = nn_pytorch.test(model, X_test)
    

if __name__ == "__main__":
    main()