import numpy as np
from scipy.stats import multivariate_normal as mvn


class K_means(object):
    """
    K means clustering algorithm.
    """
    def __init__(self):
        super(K_means, self).__init__()
        self.data = []
        self.n_centers = 0
        self.list_of_centers = []
        self.old_centers = []
        self.new_centers = []
        self.distances = []
        self.clusters = []
        self.num_samples = 0
        self.num_features = 0
        
        # raise NotImplementedError
    
    # Function for norm calculation
    def dist(self, x, y, ax=1):
        return np.linalg.norm(x - y, axis = ax)
    
    def fit(self, X, n_centers):
        """
        X: Data. A numpy array of dimensions (number of samples, number of features).
        n_centers: The number of centers. Integer.
        :return: Nothing.
        """
        self.data = X
        self.n_centers = n_centers
        self.num_samples, self.num_features = X.shape
        mean = np.mean(X, axis = 0)
        sigma = np.std(X, axis = 0)
        
        ### Initialize centers (random_data) * sigma + mean
        np.random.seed(0)
        self.list_of_centers = np.random.randn(self.n_centers, self.num_features)
        self.list_of_centers = self.list_of_centers * sigma + mean
        self.old_centers = np.zeros((self.n_centers, self.num_features))
        self.new_centers = self.list_of_centers[:]
        self.clusters = np.zeros(self.num_samples)

    def get_centers(self):
        """
        :return: Centers. A numpy array of dimensions (number of centers, number of features).
        """

        error = self.dist(self.new_centers, self.old_centers)
        
        while error.all() != 0:
            for i in range(self.num_samples):
                distances = self.dist(self.data[i], self.list_of_centers)
                cluster = np.argmin(distances)
                self.clusters[i] = cluster
            self.old_centers = self.new_centers[:]
            
            for i in range(self.n_centers):
                self.new_centers[i] = np.mean(self.data[self.clusters == i], axis = 0)
            error = self.dist(self.new_centers, self.old_centers)
        
        return np.array(self.new_centers)



class EM(object):
    """
    Expectation Maximization algorithm.
    """
    def __init__(self):
        super(EM, self).__init__()
        self.X = np.zeros((4000, 4))
        self.num_centers = 0
        self.num_samples = 0
        self.num_features = 0
        self.respons = np.zeros((4000, 4))
        # self.pis = np.array([0.25, 0.25, 0.25, 0.25])              #(4)
        # self.mus = np.array([0, 0, 1, 1])              #(4,3)
        # identity_matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        # self.sigmas = np.array([identity_matrix, identity_matrix, identity_matrix, identity_matrix])           #(4,3,3)
        self.pis = np.zeros(4)
        self.mus = np.zeros((4, 4))
        self.sigmas = np.zeros((4, 4, 4))
        self.Nks = np.zeros(4)
        self.epsilon = 0.0001
        self.max_iters = 1000
    
    
    ## E-Step: Calculate the responsiblities for all combination of centers and data points
    def expectation(self, n_centers):
        for i in range(n_centers):
            self.respons[:, i] = self.pis[i] * mvn.pdf(self.X, self.mus[i], self.sigmas[i])
        sum_of_row = np.sum(self.respons, axis = 1)
        for i in range(self.num_samples):
            self.respons[i, :] = self.respons[i, :] * (1.0/sum_of_row[i]) #Division by zero?
    
    ## M-Step: Calculate pi, mu, and sigma for each gaussian distribution
    def maximization(self, n_centers):
        self.Nks = np.sum(self.respons, axis = 0)
        for i in range(n_centers):
            self.pis[i] = self.Nks[i] / float(self.num_samples)
            self.mus[i] = (1.0 / self.Nks[i]) * np.sum(self.respons[:, i] * self.X.T, axis = 1)
            self.sigmas[i] = (1.0 / self.Nks[i]) * np.dot(np.multiply((self.X - self.mus[i]).T, self.respons[:, i].T), self.X - self.mus[i])
            # sigma_base = np.zeros((self.num_features, self.num_features))
            # for j in range(self.num_samples):
                # x_minus_mu = self.X[j] = self.mus[i]
                

    def fit(self, X, n_centers):
        """
        X: Data. A numpy array of dimensions (number of samples, number of features).
        n_centers: The number of centers. Integer.
        :return: Nothing.
        """
        self.X = X
        self.num_centers = n_centers
        self.num_samples, self.num_features = X.shape
        
        ######### Initialization of pi, mu, and sigma + responsibilites and Nks##########
        self.pis = [(1.0/n_centers) for i in range(n_centers)]
        np.random.seed(0)
        self.mus = np.random.randn(n_centers, self.num_features)
        # print(self.mus.shape)
        self.mus = X[np.random.choice(self.num_samples, n_centers), :]
        # print(self.mus.shape)
        sigmas = []
        for i in range(n_centers):
            randm = np.random.randn(self.num_features, self.num_features)
            sigmas.append(np.dot(randm.T, randm))
        self.sigmas = np.array(sigmas)
        self.sigmas = [np.eye(self.num_features) for i in range(n_centers)]
        self.respons = np.zeros((self.num_samples, n_centers))
        self.Nks = np.zeros(n_centers)
        ################################################################################
        
        ###################EM algorithm##################3
        next_L = np.inf
        for i in range(self.max_iters):
            prev_L = next_L
            self.expectation(n_centers)  # E step
            self.maximization(n_centers) # M step
            
            next_L = np.sum(np.log(np.sum(self.respons)))
            if abs(next_L - prev_L) < self.epsilon:
                break
        ###################################################

    def get_params(self):
        """
        :return: 3 gaussian mixture model parameters, pis, mus, sigmas.
                pis : A numpy array of dimensions (number of centers,).
                mus : A numpy array of dimensions (number of centers, number of features).
                sigmas : A numpy array of dimensions (number of centers, number of features, number of features).
        """
        return np.array(self.pis), np.array(self.mus), np.array(self.sigmas)
        
