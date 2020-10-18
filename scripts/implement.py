# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from helper_for_implement import *


# -*- Gradient Descent -*- #

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = []
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        ws.append(w)
        gradient = calculate_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
      
        w = w - gamma*gradient
        
        # store w and loss
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            #  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses[-1], ws[-1]
#-------------------------------------------------------#

# -*- Stochastic Gradient Descent SGD-*- #

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = []
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        ws.append(w)
        y_shuffle, tx_shuffle, stoch_gradient = compute_stoch_gradient(y, tx, w, batch_size)
        loss = compute_loss(y_shuffle, tx_shuffle, w)
        
        w = w - gamma * stoch_gradient
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # converge
            
    return losses[-1], ws[-1]
#-------------------------------------------------------#

# -*- Least Square-*- #

def least_squares(y, tx):
    """calculate the least squares."""
    opt_w = np.linalg.inv((tx.T @ tx)) @ tx.T @ y
    loss = calculate_mse(y,tx,w)
    
    return loss, opt_w

#-------------------------------------------------------#


# -*- Ridge Regression-*- #
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    opt_w = np.linalg.inv(tx.T @ tx + 2 * len(y) * lambda_ * np.eye(tx.shape[1])) @ tx.T @ y
    loss = calculate_mse(y,tx,w)
    return loss, opt_w

#-------------------------------------------------------#


# -*- Logistic Regression-*- #
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    threshold = 1e-8
    losses = []
    ws = []
    w = initial_w
    
    for iter in range(max_iter):
        # get loss and update w.
        loss = calculate_logistic_loss(y, tx, w)
        ws.append(w)
        losses.append(loss)
        gradient = calculate_logistic_gradient(y, tx, w)
        w = w - gamma * gradient
        if (len(losses) > 1) and (np.abs(losses[-1] - losses[-2]) < threshold):
            break
    return ws[-1], losses[-1]
#-------------------------------------------------------#


# -*- Regularized Logistic Regression-*- #
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    threshold = 1e-8
    losses = []
    ws = []
    w = initial_w
    
    for iter in range(max_iter):
        # get loss and update w.
        loss = calculate_penal_logistic_loss(y, tx, w, lambda_)
        gradient = calculate_penal_logistic_gradient(y, tx, w, lambda_)
        ws.append(w)
        losses.append(loss)
        w = w - gamma * gradient
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return ws[-1], losses[-1]

#-------------------------------------------------------#

