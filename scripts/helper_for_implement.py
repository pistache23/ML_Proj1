# -*- coding: utf-8 -*-
"""some helper functions for implement.py"""
import csv
import numpy as np

# -*- Function Name -*- #

#-------------------------------------------------------#


# -*- Calculate Error -*- #
def calculte_error(y,tx,w):
    e = y - tx.dot(w)
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
def compute_stoch_gradient(y, tx, w, batch_size):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    batch_size=100
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        y_shuffle = minibatch_y
        tx_shuffle = minibatch_tx
    stoch_gradient = calculate_gradient(y_shuffle, tx_shuffle, w)
    
    return y_shuffle, tx_shuffle, stoch_gradient
#-------------------------------------------------------#



# -*- Build Polynomial Extension -*- #
def build_poly(x, degree):
    """expand the target features(n) into 1+degree*n_features degrees
    such as: x_1.T = [2,3] ==> x_1.T = [1,2,4,3,9] if degree = 2
    """
    
    tx = np.ones([len(x),1])
    for col in range(x.shape[1]):
        for i in range(degree):
            tx = np.hstack([tx, x[:,col][:,np.newaxis] ** (i+1)])
    return tx
#-------------------------------------------------------#


# -*- Calculate sigmoid -*- #
def sigmoid(t):
    return 1. / (1. + np.exp(-t))
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
    return loss[0][0]
#-------------------------------------------------------#

# -*- Calculate Penalized Logistic Gradient -*- #
def calculate_penal_logistic_gradient(y, tx, w, lambda_):
    gradient = calculate_logistic_gradient(y, tx, w) + lambda_ * w
    return gradient
#-------------------------------------------------------#



# -*- Cross Validation -*- #

def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree,ridge): # if ridge=True, this is used to test ridge regression, otherwise this is used to test reg-logistic regression
    """return the loss of ridge regression."""
    
    y_test = y[k_indices[k,:]]
    x_test = x[k_indices[k,:]]
    
    k_indicess = np.delete(k_indices, k, axis = 0) #to choose the other (k-1) rows
    y_train = y[k_indicess]
    x_train = x[k_indicess]
    
    # to transfer the dimension into 2 dimentions
    y_train =  y_train.reshape((4*y_train.shape[1]))
    x_train = x_train.reshape((4*x_train.shape[1],x_train.shape[2]))

    if degree > 0:
        x_test = build_poly(x_test, degree)
        x_train = build_poly(x_train, degree)

    if ridge:
        _,opt_w = ridge_regression(y_train.reshape((-1,1)), x_train, lambda_)
        loss_te = calculate_rmse(y_test.reshape((-1,1)),x_test,opt_w)
        
    if not ridge:
        _,opt_w = reg_logistic_regression(y_train.reshape((-1,1)), x_train, lambda_, np.ones((x_train.shape[1],1)),3000,0.0007)
        
        yy = y_test.copy()
        yy[yy == -1] = 0
        loss_te = calculate_penal_logistic_loss(yy.reshape((-1,1)),x_test,opt_w, lambda_)
        
    return  loss_te
#-------------------------------------------------------#

# -*- Prediction -*- #

def prediction_report_4(y, tx, w_best): # to predict the first four models, except for logistic and reg-logistic regression
    
    predictions = tx @ w_best
    predictions[:][predictions >= 0] = 1
    predictions[:][predictions < 0] = -1
    
    percentage_model_4 = []
    for i in range(4):
        correct_percentage = 100*np.sum(predictions[i] == y) / float(len(predictions[i]))
        percentage_model_4.append(correct_percentage)
        
    return percentage_model_4

def change_labels_logistic(y): # to convert the -1 lable into 0 label to adapt the sigmoid districution
    y_change = y.copy()
    y_change[y_change == -1] = 0
    return y_change

def prediction_logistic_report(y, tx, w_best): #to predict the last two logistic models
    

    predictions = tx @ w_best
    predictions[:][predictions >= 0.5] = 1
    predictions[:][predictions < 0.5] = 0
    y_log = change_labels_logistic(y)
        
    percentage_model_logistic = []
    for i in range(4,6):
        correct_percentage = 100*np.sum(predictions[i] == y_log) / float(len(predictions[i]))
        percentage_model_logistic.append(correct_percentage)
        
    return percentage_model_logistic
