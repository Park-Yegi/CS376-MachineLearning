import numpy as np
import cvxopt
import math


# Problem 1: Linear Regression
class LinearRegression(object):
    """
    A linear regression model.

    Methods
    -------
    fit() : takes data and targets as input and trains the model with it.
    predict() : takes data as input and predicts the target values with the trained model.
    """
    param_beta = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.param_beta = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def fit(self, X, y):
        """
        This function should train the model. By invoking this function, a class member variable
        containing the coefficients of the model will be filled based on the input data. These coefficients
        will be used in the predict() function to make predictions about unknown data.

        X: Data. A numpy array of dimensions (number of samples, number of features). These are real values.
        y: Targets. A numpy array of dimensions (number of samples,). These are real values.
        :return: The mean squared error for the input training data.
        """
        B = np.linalg.pinv(X)  #Compute Moore-Penrose pseudoinverse matrix B
        self.param_beta = np.matmul(B, y)
        
        ## Calculate mean squared error for the input training data
        sum_of_squared_error = 0
        for i in range(0, 400):
            sum_of_squared_error += (y[i] - np.dot(X[i], self.param_beta))**2
        
        mean_squared_error = sum_of_squared_error / 400
        return mean_squared_error

    def predict(self, X):
        """
        This function should predict the target values using the parameters learned with the fit() function given the
        input test data.

        X: Data. A numpy array of dimensions (number of samples, number of features).

        :return: A numpy array of dimensions (number of samples,) that contains predicted targets for the input data X.
        """
        y = np.matmul(X, self.param_beta)
        return y
        # raise NotImplementedError

# Problem 2: Logistic Regression
class LogisticRegression(object):
    """
    A logistic regression model.

    Methods
    -------
    fit() : takes data and targets as input and trains the model with it.
    predict() : takes data as input and predicts the target values with the trained model.
    """
    ## Define hyperparameter (learning rate, number of iterations)
    learning_rate = 0
    iterations = 0
    beta = np.array([0,0,0,0,0,0,0,0,0,0])
    
    def __init__(self, learning_rate=0.001, iterations=450):
        """
        This is one way of hard coding the hyperparameters you found. You dont have to follow this conventions.
        However, calling LogisticrRegression() without any parameters must create a model loaded with your best hyperparameters.
        """
        super(LogisticRegression, self).__init__()
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.beta = np.array([0,0,0,0,0,0,0,0,0,0])
    
    # Define Logistic function, x: array[10]
    def sigmoid(self, x): 
        return float(1) / (1.0 + np.exp(np.dot(self.beta, x)*(-1)))
    
    # Calculate the direction of gradient descent by taking the gradient of loss function
    def gradient_of_loss(self, X, y):
        result = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        for i in range(0, 100):
            result += (self.sigmoid(X[i]) - y[i]) * X[i]
        return result
    
    # calculate cost function: -sum(y*log(p)+(1-y)*log(1-p))
    def loss_func(self, X, y):
        sum_of_cost = 0
        for i in range(0, 100):
            if y[i] == 1:
                sum_of_cost += math.log(self.sigmoid(X[i]))
            else:
                sum_of_cost += math.log(1.0-self.sigmoid(X[i]))
        return (sum_of_cost * (-1.0))
    
    def fit(self, X, y):
        """
        This function should train the model. By invoking this function, a class member variable
        containing the coefficients of the model will be filled based on the input data. These coefficients
        will then be used in the predict() function to make predictions about unknown data.

        X: Data. A numpy array of dimensions (number of samples, number of features). These are real values.
        y: Targets. A numpy array of dimensions (number of samples,). These are discrete values(either 0 or 1).
        :return: A list containing loss values for each iteration.
        """
        # Gradient Descent
        loss_iter = []
        loss = self.loss_func(X, y)
        loss_iter.append(loss)
        for i in range(0, self.iterations):
            self.beta = self.beta - (self.learning_rate * self.gradient_of_loss(X, y))
            loss = self.loss_func(X, y)
            loss_iter.append(loss)
        return np.array(loss_iter)

    def predict(self, X):
        """
        This function should predict the target values using the parameters learned with the fit() function given the
        input data.

        :param X: Data. A numpy array of dimensions (number of samples, number of features).
        :return: A numpy array of dimensions (number of samples,) that contains
        predicted targets for the input data X. These targets must be discrete: either 0 or 1.
        """
        pred_target = []
        for i in range(0, np.size(X, 0)):
            pred_prob = self.sigmoid(X[i])
            if pred_prob >= 0.5:
                pred_target.append(1)
            else:
                pred_target.append(0)
        return np.array(pred_target)

# Problem 3: Support Vector Machine
class SVM(object):
    """
    A support vector machine with RBF kernel.

    Methods
    -------
    fit() : takes data and targets as input and trains the model with it.
    predict() : takes data as input and predicts the target values with the trained model.
    """
    C = 0.0
    sigma = 0.0
    alpha = []
    w = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    b = 0.0
    training_y = []
    training_X = []
    M = 0
    support_vecs = []
    
    def __init__(self, C=4.0, sigma=2.0):
        """
        This is one way of hard coding the hyperparameters you found. You dont have to follow this conventions.
        However, calling LogisticrRegression() without any parameters must create a model loaded with your best hyperparameters.
        """
        super(SVM, self).__init__()
        self.C = C
        self.sigma = sigma
        self.w = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.b = 0.0
        self.M = 500 # number of data points in training set
        self.alpha = [0 for i in range(self.M)]
    
    # square of l2 norm
    def l2_norm(self, x):
        return np.dot(x, x)
    
    # Function for Gaussian kernel
    def RBF_kernel(self, x, y):
        temp1 = self.l2_norm(x-y) * (-1)
        temp2 = 2*(self.sigma**2)
        return np.exp(temp1 / temp2)
    
    # Solve Quadratic Programming(dual problem) with cvxopt library
    def solve_QP(self, X, y):
        P = cvxopt.matrix(np.array([[(y[i]*y[j]*self.RBF_kernel(X[i], X[j])) for i in range(self.M)] for j in range(self.M)]))
        q = cvxopt.matrix(-np.ones(self.M))
        G = cvxopt.matrix(np.concatenate((-np.eye(self.M), np.eye(self.M)), axis=0))
        h = cvxopt.matrix(np.concatenate((np.zeros(self.M), self.C*np.ones(self.M)), axis=0))
        A = cvxopt.matrix(y, (1, self.M))
        b = cvxopt.matrix(0.0)
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alpha = np.array(sol['x']).reshape(self.M)
        # print(self.alpha)
        return self.alpha
    
    def fit(self, X, y):
        """
        This function should train the model. By invoking this function, a class member variable
        containing the coefficients of the model will be filled based on the input data. These coefficients
        will then be used in the predict() function to make predictions about unknown data.

        :param X: Data. A numpy array of dimensions (number of samples, number of features).
        :param y: Targets. A numpy array of dimensions (number of samples,). These are discrete values(either 0 or 1).
        :return: Not important.
        """
        self.training_y = [(i*2)-1 for i in y]
        self.training_X = X
        
        self.solve_QP(X, self.training_y)

        for i in range(self.M):
            if self.alpha[i] > 1e-7:
                self.support_vecs.append(i)
                    
        list_b = []
        for i in self.support_vecs:
            edge_b = self.training_y[i]
            for j in range(self.M):
                edge_b -= self.alpha[j]*self.training_y[j]*self.RBF_kernel(X[i], X[j])
            list_b.append(edge_b)
        if len(list_b) != 0:
            self.b = sum(list_b) / len(list_b)
        else: 
            self.b = 0
        

    def predict(self, X):
        """
        This function should predict the target values using the parameters learned with the fit() function given the
        input data.

        :param X: Data. A numpy array of dimensions (number of samples, number of features).
        :return: A numpy array of dimensions (number of samples,) that contains
        predicted targets for the input data X. These targets must be discrete: either 0 or 1.
        """
        pred_target = []
        for i in range(0, np.size(X, 0)):
            pred_value = 0
            for j in self.support_vecs:
                pred_value += self.alpha[j] * self.training_y[j] * self.RBF_kernel(self.training_X[j], X[i])
            pred_value += self.b

            if pred_value >= 0:
                pred_target.append(1)
            else:
                pred_target.append(0)
        return np.array(pred_target)
