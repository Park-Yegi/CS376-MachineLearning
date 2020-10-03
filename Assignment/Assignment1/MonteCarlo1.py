import numpy as np

def main():
    num_list = [100, 500, 1000, 10000, 100000]
    result = []

    for num in num_list:
        count = 0
        np.random.seed(12345)
        rand_x = np.random.rand(num)
        np.random.seed(54321)
        rand_y = np.random.rand(num)
        for i in range(num):
            if (rand_x[i] ** 2 + rand_y[i] ** 2 < 1):
                count += 1
        pi = (count / num) * 4
        result.append(pi)
        print(pi)
    return result

if __name__ == "__main__":
    main()
