# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

# define function to split data according to jet_num
def split_data(y, tx, ids, jet_num):
    mark = tx[:,22] == jet_num
    return y[mark], tx[mark], ids[mark]


# to analyze the columns variance 
def variance_analysis(tx): # to get the zero-variance features (to be deleted) 
    cols = tx.shape[1]
    deleted_columns = []
    for col in range(cols):
        target_col = tx[:,col]
        if np.var(target_col) < 10**-3:
            deleted_columns.append(col)
    return deleted_columns

# to delete the zero-variance columns
def delete_columns(tx, deleted_columns, features):
    delete_num = 0
    for col in deleted_columns:
        tx = np.delete(tx, col - delete_num, 1)
        features = np.delete(features, col - delete_num)
        delete_num += 1
    return tx, features


def fix_null(tx):
    col_num = tx.shape[1]
    for col in range(col_num):
        current_col = tx[:, col]
        current_col[current_col == -999] = \
        np.median(current_col[current_col != -999])
        
def fix_outlier(tx):
    col_num = tx.shape[1]
    for col in range(col_num):
        current_col = tx[:, col]
        std = np.std(current_col)
        mean = np.mean(current_col)
        left_boundary = mean - 2 * std
        right_boundary = mean + 2 * std
        current_col[current_col < left_boundary] = left_boundary
        current_col[current_col > right_boundary] = right_boundary        
 
        
def normalization(x):
    min_x = np.min(x, axis=0)
    max_x = np.max(x, axis=0)
    return (x - min_x) / (max_x - min_x)
