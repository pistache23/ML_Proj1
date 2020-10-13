# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

# -*- Cost Function -*- #
def compute_loss(y, tx, w):
    
    e = y[:,np.newaxis] - tx.dot(w)
    MSE = 0.5/len(y)*np.sum(e**2)
    return MSE

    raise NotImplementedError
#-------------------------------------------------------#


# -*- Gradient Descent -*- #
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y[:,np.newaxis] - tx.dot(w)
    gradient = -1/len(y) * tx.T.dot(e)
    
    return gradient
    raise NotImplementedError


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = []
    losses = []
    w = initial_w.reshape((-1,1))
    
    for n_iter in range(max_iters):
        ws.append(w)
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
      
        w = w - gamma*gradient
        
        # store w and loss
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            #  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
#-------------------------------------------------------#

# -*- Gradient Descent SGD-*- #
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


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        y_shuffle = minibatch_y
        tx_shuffle = minibatch_tx
    stoch_gradient = compute_gradient(y_shuffle, tx_shuffle, w)
    
    return stoch_gradient
    raise NotImplementedError


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = []
    losses = []
    w = initial_w.reshape((-1,1))
    
    for n_iter in range(max_iters):
        ws.append(w)
        stoch_gradient = compute_stoch_gradient(y, tx, w, batch_size)
        loss = compute_loss(y_shuffle, tx_shuffle, w)
        
        w = w - gamma * stoch_gradient
        # store w and loss
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    #print("minimum loss:{a:.6f} optimal W*:{b}".format(a = np.min(losses), b = ws[np.argmin(losses)]))
   # print(np.argmin(losses))
    return losses, ws
    
    raise NotImplementedError
#-------------------------------------------------------#

# -*- Least Square-*- #

def least_squares(y, tx):
    """calculate the least squares."""
    opt_w = np.linalg.inv((tx.T @ tx)) @ tx.T @ y
    MSE = 0.5 / len(y) * (y - tx @ opt_w).T @ (y - tx @ opt_w) 
    return opt_w, MSE
    raise NotImplementedError
#-------------------------------------------------------#


# -*- Ridge Regression-*- #
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    opt_w = np.linalg.inv(tx.T @ tx + 2 * len(y) * lambda_ * np.eye(tx.shape[1])) @ tx.T @ y
    return opt_w
    raise NotImplementedError
#-------------------------------------------------------#


# -*- Logistic Regression-*- #


#-------------------------------------------------------#


# -*- Regularized Logistic Regression-*- #


#-------------------------------------------------------#

