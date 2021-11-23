import os
import json
import numpy as np
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, pearsonr, ttest_ind, spearmanr
from scipy.interpolate import interp1d, PchipInterpolator
from glob import glob
from datetime import datetime
from IPython.display import clear_output
import random
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
import tensorflow.keras.backend as K
from B_dataprocessing import *
from B_DNN_model import *
from B_plotting import *

class battery_model():
    def __init__(self,i,X,y,Y,X_Y_merged,objective):
        self.group = i
        self.X = X
        self.originalx = X
        self.y = y
        self.Y = Y
        self.X_Y_Merge = X_Y_merged
        self.objective = objective
        #generate y with key
        y_matrix_key = np.array([k for k,v in Y.items()]).reshape(-1,1)
        self.y_selected_key = y_matrix_key.squeeze()
        self.y_with_key = np.append(y, y_matrix_key, axis = 1)
        self.model = None
        self.y_pred = None
        self.var_num = None
        self.lr_model_history = None
        self.error = None
        self.test_key = None
        self.y_with_key_test = None
        self.y_with_key_test_true = None

    def plot_y_point(self,norm=False):
        '''
        If norm = True
        Plot the y after normalization w.r.p to each class
        '''
        x_ax = range(len(self.X))
        _,num_outputs = self.y.shape
        for i in range(num_outputs):
            if norm == False:
                plt.scatter(x_ax, self.y[:,i],  s=6, label=f"y{i}-test")
            else:
                plt.scatter(x_ax, (self.y[:,i]-min(self.y[:,i]))/(max(self.y[:,i])-min(self.y[:,i])),  s=6, label=f"y{i}-test-norm")
        plt.legend()
        plt.show()

    def plot_curve(self, option,all_metrics_df,l=1000,d=1):
        '''
        option = 'all' -> plot all exprimental data
        option = 'selected" -> plot filtered ones
        option = 'test' -> plot predicted ones
        '''
        if option == 'all':
            
            plt.figure(figsize=(9,6))
            plot_empirical(l,d,self.all_metrics_df)
        elif option == 'selected':
            plot_empirical_selected(self.y_selected_key,l,d,all_metrics_df)
        elif option == 'test':
            plot_curve_pred(self.y_with_key_test,self.test_key,l,d,all_metrics_df,self.objective)
        elif option == 'test_true':
            plot_curve_pred(self.y_with_key_test_true,self.test_key,l,d,all_metrics_df,self.objective)
        elif option == 'fitting':
            plot_curve_pred(self.y_with_key,self.y_selected_key,l,d,all_metrics_df,self.objective)
#预测

    def input_argumentation_update(self,k):
        X_original = self.X.copy()
        self.X = create_poly(X_original,k)
        
    def input_restore(self):
        self.X = self.originalx

    def model_construction_and_training(self,all_metrics_df,loss_fun,class_weight,nl,opt,epochs, verbose):
        '''
        Parameters: X and y_with_key
        Hyperparameters:
        1. loss_fun : 'mae' or 'mse'
        2. class_weight: default = uniform should be dictionary of 3 classes
            class_weight0 = {0: 1.,
                            1: 1.,
                            2: 1.}
        3. number of neouron for each layer [] (2 to 4)
        4. epoches
        5. verbose
        
        output: model0,  defined evaluation matrics, (graph of prediction ,)
        '''

        X_train,X_test,X_val, y_train_coef,y_test_coef,y_val_coef,y_train,y_test,y_val,y_train_norm,y_test_norm,y_val_norm=data_split(self.X,self.y_with_key)
        self.model = sequential_model(loss_fun,X_train,y_train,nl,opt)

        lr_model_history = self.model.fit(X_train, y_train_norm, 
                                epochs=epochs, 
                                verbose=verbose,
                                class_weight = class_weight
                                ,validation_data=(X_val, y_val_norm))
        self.lr_model_history = lr_model_history
        
        raw_y_pred = self.model.predict(X_test)
        self.y_pred = recover(raw_y_pred,y_train)
        self.var_num = self.y_pred.shape[1]

        y_pred_true= recover(y_test_norm,y_train)

        test_key = y_test_coef[:,-1]
        self.test_key = test_key
        self.y_with_key_test = np.append(self.y_pred, test_key.reshape(-1,1), axis = 1)
        self.y_with_key_test_true = np.append(y_pred_true, test_key.reshape(-1,1), axis = 1)

        self.error = error_after_plot(self.y_with_key_test,self.y_with_key_test_true,test_key,all_metrics_df,self.objective)
        return self.error


    def loss_function_plot(self):
        # Plot the loss function with no val
        fig, ax = plt.subplots(1, 1, figsize=(10,6))
        ax.plot(np.sqrt(self.lr_model_history.history['loss']), 'r', label='train')
        ax.set_xlabel(r'Epoch', fontsize=20)
        ax.set_ylabel(r'Loss', fontsize=20)
        ax.legend()
        ax.tick_params(labelsize=20)

    