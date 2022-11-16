# coding=utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import numpy as np
from numpy import random


def main():
    features = 5
    test_amount = 80
    beta = [[10], [8], [12], [9], [6]]
    x = np.random.rand(test_amount, features)
    for i in np.arange(0, 3, 0.5):
        b = linear_reg(x, i, beta, test_amount)
        print(b.transpose())


def linear_reg(x, sigma, beta, test_amount):
    noise = random.normal(loc=0, scale=sigma, size=(test_amount, 1))
    y = np.matmul(x, beta) + noise
    x_t = x.transpose()
    x_t_x = np.matmul(x_t, x)
    x_t_x_inv = np.linalg.inv(x_t_x)
    x_t_y = np.matmul(x_t, y)
    beta_new = np.matmul(x_t_x_inv, x_t_y)
    return beta_new


if __name__ == "__main__":
    main()
