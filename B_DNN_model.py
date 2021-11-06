from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import json
import numpy as np
import pickle
import time
import pandas as pd

def norm (x,x_train):
    return (x-x_train.min(axis=0))/(x_train.max(axis=0)-x_train.min(axis=0))
def recover(x,x_train):
    return x*(x_train.max(axis=0)-x_train.min(axis=0))+x_train.min(axis=0)

def data_split(X,y_with_key,i=3):
    X_copy = X.copy()
    y_with_key_copy = y_with_key.copy()
    X_train, X_test, y_train_coef, y_test_coef = train_test_split(X_copy, y_with_key_copy, test_size=0.2, random_state=3)
    y_train = y_train_coef[:,0:i]
    y_test = y_test_coef[:,0:i]

    y_train_norm = norm(y_train,y_train)
    y_test_norm =  norm(y_test,y_train)
    return X_train,X_test,y_train_coef,y_test_coef,y_train,y_test,y_train_norm,y_test_norm

def sequential_model(l1,l2,loss_method,X_train,y_train):
    in_dim= X_train.shape[1]
    out_dim= y_train.shape[1]
    model0 = Sequential([
    Dense(l1, input_dim=in_dim, activation="relu"),
    Dense(l2, activation="relu"),
    Dense(out_dim)
    ])
    model0.compile(loss=loss_method,
                    optimizer="adam",
                    metrics = ['accuracy'])
    return model0
