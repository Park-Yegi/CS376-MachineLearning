import numpy as np


class MLP(object):
    """
    Multi Layer Perceptron model.
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.X_train = []
        self.y_train = []
        self.w1 = np.random.randn(2, 100) / np.sqrt(2)
        # self.b1 = np.random.randn(1, 100)   #w1 and b1 for  input data --> hidden layer
        self.w2 = np.random.randn(100, 1) / np.sqrt(100)
        # self.b2 = np.random.randn(1, 1)     #w2 and b2 for hidden layer --> output layer
        self.hidden_layer = np.zeros(100)
        self.output_layer = 0
        self.y_hat = np.zeros(1000)
        self.n_layers = 3    #include input layer, hidden layer, and output layer
        self.n_hidden_nodes = 100
        self.loss = None
        self.learning_rate = 0.0002
        self.epsilon = 1e-4
        self.iterations = 3000
        self.output_error = 0
        self.output_delta = 0
        self.hidden_error = 0
        self.hidden_delta = 0
        
    
    def ReLU(self, x):
        return np.maximum(0, x)
            
    def ReLU_derivative(self, x):
        return 1.0 * (x > 0)

        
    def feed_forward(self, x):
        # a = np.dot(x, self.w1) + self.b1
        a = np.dot(x, self.w1)
        self.hidden_layer = self.ReLU(a)
        # self.hidden_layer = [self.ReLU(x) for x in a]
        # b = np.dot(h, self.w2) + self.b2
        b = np.dot(self.hidden_layer, self.w2)
        self.output_layer = self.ReLU(b)
        return a, b


    def back_propagation(self, a, b):
        # print(self.output_layer.shape)   (1000, 1)
        # print(b.shape)    (1000, 1)
        temp1 = np.multiply(-(self.y_train-self.output_layer), self.ReLU_derivative(b))
        # print(temp1.shape)     (1000, 1)
        # print(self.w2.shape)      (100, 1)
        temp2 = np.multiply(np.dot(temp1, self.w2.T), self.ReLU_derivative(a))
        # print(np.dot(temp1, self.w2.T).shape)     (1000, 100)
        # print(a.shape)         (1000, 100)
        # print(temp2.shape)     (1000, 100)
        
        # self.output_error = y - self.output_layer
        # self.output_delta = self.output_error * self.ReLU_derivative(self.output_layer)  
        
        # self.hidden_error = self.output_delta.dot(self.w2.T)   # dimension: (100, )
        # derivative_hidden_layer = [self.ReLU_derivative(x) for x in self.hidden_layer]  # dimension: (100, )
        # self.hidden_delta = self.hidden_error * self.ReLU_derivative(self.hidden_layer)
        
        # # temp = self.output_error.dot(self.w2.T) * derivative_hidden_layer
        delta_w1 = np.dot(self.X_train.T, temp2)
        # # delta_b1 = 
        delta_w2 = np.dot(self.hidden_layer.T, temp1)
        # # delta_b2 = 
        
        # return delta_w1, delta_b1, delta_w2, delta_b2
        # delta_w1 = np.zeros((2, 100))
        # delta_w2 = np.zeros((100, 1))
        return delta_w1, delta_w2
        
    
    def fit(self, X, y):
        """
        X: Data. A numpy array of dimensions (number of samples, number of features).
        y: Targets. A numpy array of dimensions (number of samples,).
        :return: Nothing.
        """
        self.X_train = X
        self.y_train = y
        
        for i in range(self.iterations):
            a, b = self.feed_forward(X)
#             # print(self.output_layer)
            # loss = sum((y-self.output_layer)**2) /2

            # delta_w1, delta_b1, delta_w2, delta_b2 = self.back_propagation(y[i])
            delta_w1, delta_w2 = self.back_propagation(a, b)
            # self.back_propagation(a, b)

            self.w1 -= self.learning_rate * delta_w1
            # self.b1 -= self.learning_rate * delta_b1
            self.w2 -= self.learning_rate * delta_w2
            # self.b2 -= self.learning_rate * delta_b2
        # print(np.array(self.output_layer))
            

    def predict(self, X):
        """
        X: Data. A numpy array of dimensions (number of samples, number of features).
        :return: Predicted targets. A numpy array of dimensions (number of samples,)
        """
        self.X_test = X
        prediction = []
        self.feed_forward(X)
        # print("Print Prediction for nn_test")
        print(np.array(self.ReLU_derivative(self.output_layer)))
        # print(np.array(self.output_layer).shape)
        
        return np.array(self.ReLU_derivative(self.output_layer))
        # return np.array([0,1,0,1,0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,1,0,0,0,1,1,1,1,1,1,0,1,1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,1,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,0,0,1,0,0,1,1,0,0,1,1,0,1,0])
        