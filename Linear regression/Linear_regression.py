import numpy as np
from numba import njit

X_vect = np.array([[1,2],[3,4],[4,5]])
theta_vect = np.array([5, 6])
y_vect = np.array([1, 2, 1])

@njit
def linear_func(theta, x):
    return (x*theta).sum()

@njit
def linear_func_all(theta, X):
    return np.array([linear_func(theta, x) for x in X])

@njit
def mean_squared_error(theta, X, y):
    M = len(X)
    return 1/M * np.array([(y[i] - linear_func_all(theta, X)[i])**2 for i in range(X.shape[0])]).sum()

@njit
def grad_mean_squared_error(theta, X, y):
    h = linear_func_all(theta, X)
    M = X.shape[0]
    return np.array([2/M * ((y - h) * (-X.transpose()[i])).sum() for i in range(len(theta))])

a = linear_func_all(theta_vect, X_vect)

b = mean_squared_error(theta_vect, X_vect, y_vect)

c = grad_mean_squared_error(theta_vect, X_vect, y_vect)

print(f"{a}\n{b}\n{c}")