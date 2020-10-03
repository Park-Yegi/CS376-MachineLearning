import numpy as np
# import matplotlib.pyplot as plt
# import cvxopt
# from models import LinearRegression
# from models import SVM

# from elice_utils import EliceUtils

# elice_utils = EliceUtils()


def main():
    """
    This is the main function. Use this area as you like.
    Below is an example of a simple workflow. Get the data, train the model,
    predict based on the model, then evaluate the performance.
    The code is just for demonstration, you can just erase it and make your own.
    """
    import numpy as np
    # from models import LinearRegression
    # from models import SVM

    data = np.loadtxt("svm_10d_train.txt", " ", np.void)
    for i in range(0, 500):
        print(data[i])
    # Synthesize some data for linear regression.
    # X = np.random.uniform(-1, 1, (200, 2))
    # w = np.random.normal(0, 5, 2)
    # y = np.dot(X, w) + np.random.normal(0, 0.5, 200)
    # X_train = X[:100,:]
    # y_train = y[:100]
    # X_test = X[100:,:]
    # y_test = y[100:]
    
    
    
    # Make a model and train it.
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    # Predict some target values and evaluate the performance.
    # y_pred = model.predict(X_test)
    # mse = np.dot(y_pred - y_test, y_pred - y_test)/y_test.shape[0]
    # print(mse)
    
    # model = SVM()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # score = 0
    # for i in range(0, 100):
    #     if y_test[i] == y_pred[i]:
    #         score += 1
    # print(score)


if __name__ == "__main__":
    main()