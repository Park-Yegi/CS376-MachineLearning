import numpy as np
import pylab as pl
import scipy.special as ss
import matplotlib.pyplot as plt

def beta(a, b, mew):
    e1 = ss.gamma(a + b)
    e2 = ss.gamma(a)
    e3 = ss.gamma(b)
    e4 = mew ** (a - 1)
    e5 = (1 - mew) ** (b - 1)
    return (e1/(e2*e3)) * e4 * e5

def plot_beta(a, b):
    Ly = []
    Lx = []
    mews = np.mgrid[0.4:0.6:0.01]
    for mew in mews:
        Lx.append(mew)
        Ly.append(beta(a, b, mew))
    x_array = np.array(Lx)
    y_array = np.array(Ly)
    plt.figure()
    plt.scatter(x_array, y_array)

def main():
    plot_beta(8, 7)  #if plot_beta(6, 6) -> answer for problem7-a
    pl.xlim(0.0, 1.0)
    pl.ylim(0.0, 4.0)
    plt.show()

if __name__ == "__main__":
    main()
