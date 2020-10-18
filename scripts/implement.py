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
    threshold = 1e-8
    
    for n_iter in range(max_iters):
        ws.append(w)
        gradient = calculate_gradient(y, tx, w)
        loss = calculate_mse(y, tx, w)
      
        w = w - gamma*gradient
        
        # store w and loss
        losses.append(loss)
        if (len(losses) > 1) and (np.abs(losses[-1] - losses[-2]) < threshold):
            break 
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            #  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses[-1], ws[-1]
#-------------------------------------------------------#

# -*- Stochastic Gradient Descent SGD-*- #

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    threshold = 1e-8
    ws = []
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        ws.append(w)
        y_shuffle, tx_shuffle, stoch_gradient = compute_stoch_gradient(y, tx, w,batch_size)
        loss = calculate_mse(y_shuffle, tx_shuffle, w)
        
        w = w - gamma * stoch_gradient
        losses.append(loss)
        if (len(losses) > 1) and (np.abs(losses[-1] - losses[-2]) < threshold):
            break # converge
            
    return losses[-1], ws[-1]
#-------------------------------------------------------#

# -*- Least Square-*- #

def least_squares(y, tx):
    """calculate the least squares."""
    coefficient = tx.T.dot(tx)
    constant = tx.T.dot(y)
    
    opt_w = np.linalg.solve(coefficient, constant)
    loss = calculate_mse(y,tx,opt_w)
    
    return loss, opt_w

#-------------------------------------------------------#


# -*- Ridge Regression-*- #
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    coefficient = tx.T.dot(tx) + 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    constant = tx.T.dot(y)
    
    opt_w = np.linalg.solve(coefficient, constant)
    loss = calculate_mse(y,tx,opt_w)
    return loss, opt_w

#-------------------------------------------------------#


# -*- Logistic Regression-*- #
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    threshold = 1e-8
    losses = []
    ws = []
    w = initial_w
    yy = y.copy()
    yy[yy == -1] = 0
    for iter in range(max_iters):
        # get loss and update w.
        loss = calculate_logistic_loss(yy, tx, w)
        ws.append(w)
        losses.append(loss)
        gradient = calculate_logistic_gradient(yy, tx, w)
        w = w - gamma * gradient
        if (len(losses) > 1) and (np.abs(losses[-1] - losses[-2]) < threshold):
            break
    return losses[-1], ws[-1]
#-------------------------------------------------------#


# -*- Regularized Logistic Regression-*- #
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    threshold = 1e-8
    losses = []
    ws = []
    w = initial_w
    yy = y.copy()
    yy[yy == -1] = 0
    for iter in range(max_iters):
        # get loss and update w.
        loss = calculate_penal_logistic_loss(yy, tx, w, lambda_)
        gradient = calculate_penal_logistic_gradient(yy, tx, w, lambda_)
        ws.append(w)
        losses.append(loss)
        w = w - gamma * gradient
        
        if (len(losses) > 1) and (np.abs(losses[-1] - losses[-2]) < threshold):
            break
    return losses[-1], ws[-1]

#-------------------------------------------------------#

