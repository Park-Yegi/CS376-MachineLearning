import numpy as np
import math

def quater_circle(num):
    return math.sqrt(1-(num**2))

def main():
    num_list = [100, 500, 1000, 5000, 10000]
    result = []

    ### in Monte Carlo Integration
    ### unbiased estimator of integral is f(x)/(b-a)
    ### in this problem f(x) = sqrt(1-x**2), b=1, a=0
    for num in num_list:
        sum = 0
        np.random.seed(493)
        rand_num = np.random.rand(num)
        for i in range(num):
            sum += quater_circle(rand_num[i])
        pi = (sum / num) * 4
        result.append(pi)
        print(pi)
    return result

if __name__ == "__main__":
    main()
