# -*- coding: utf-8 -*-
"""some helper functions for implement.py"""
import csv
import numpy as np

# -*- Function Name -*- #

#-------------------------------------------------------#


# -*- Calculate Error -*- #
def calculte_error(y,tx,w):
    e = y[:,np.newaxis] - tx.dot(w)
    return e
#-------------------------------------------------------#


# -*- Calculate MSE -*- #
def calculate_mse(y,tx,w):
    e = calculte_error(y,tx,w)
    mse = 0.5 * np.mean(e ** 2) 
    return mse
#-------------------------------------------------------#

# -*- Calculate RMSE -*- #
def calculate_rmse(y,tx,w):
    mse = calculate_mse(y,tx,w)
    rmse = np.sqrt(2 * mse)
    return rmse
#-------------------------------------------------------#

# -*- Calculate Gradient -*- #
def calculate_gradient(y, tx, w):
    e = calculte_error(y,tx,w)
    gradient = -1/len(y) * tx.T.dot(e)
    return gradient
#-------------------------------------------------------#

# -*- Build MiniBatch -*- #
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
#-------------------------------------------------------#

# -*- Calculate Stochastic Gradient -*- #
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        y_shuffle = minibatch_y
        tx_shuffle = minibatch_tx
    stoch_gradient = compute_gradient(y_shuffle, tx_shuffle, w)
    
    return y_shuffle, tx_shuffle, stoch_gradient
#-------------------------------------------------------#



# -*- Build Polynomial Extension -*- #
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    tx = np.ones([len(x),1])
    for i in range(degree):
        tx = np.hstack([tx, x[:,np.newaxis] ** (i+1)])
    return tx
#-------------------------------------------------------#


# -*- Calculate sigmoid -*- #
def sigmoid(t):
    return np.exp(t) / (1 + np.exp(t))
#-------------------------------------------------------#


# -*- Calculate Logistic Loss -*- #
def calculate_logistic_loss(y, tx, w):
    loss = np.sum(np.log(1. + np.exp(tx @ w)) - y * tx @ w)
    return loss
#-------------------------------------------------------#

# -*- Calculate Logistic Gradient -*- #
def calculate_logistic_gradient(y, tx, w):
    gradient = tx.T @ (sigmoid(tx @ w) - y)
    return gradient
#-------------------------------------------------------#

# -*- Calculate Penalized Logistic Loss -*- #
def calculate_penal_logistic_loss(y, tx, w, lambda_):
    loss = calculate_logistic_loss(y, tx, w) + (lambda_ / 2) * (w.T @ w)
    return loss
#-------------------------------------------------------#

# -*- Calculate Penalized Logistic Gradient -*- #
def calculate_penal_logistic_gradient(y, tx, w, lambda_):
    gradient = calculate_logistic_gradient(y, tx, w) + lambda_ * w
    return gradient
#-------------------------------------------------------#