import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator,HillClimbSearch
from pgmpy.estimators.StructureScore import K2Score, BDeuScore, BDsScore, BicScore
#!pip install seaborn

def plot_empirical(l,d,all_metrics_df):
    for i in range(328):
        for_one_cell = all_metrics_df.loc[(all_metrics_df.seq_num == list(all_metrics_df.seq_num.unique())[i])] # get all cycling data for 1 cell
        if len(for_one_cell) > 3:  # not select cells have too less data point
            y = for_one_cell['diag_discharge_capacity_rpt_0.2C']
            #x = for_one_cell['cycle_index']
            x = for_one_cell['equivalent_full_cycles']
            #y1 =  y / float(y.head(1)) 
            plt.plot(x,y)
            plt.xlabel("Equivalent Full Cycle")
            plt.ylabel("Capacity")
            plt.title("Capacity degradation",fontsize=16)
            plt.xlim(0,l)
            plt.ylim(d,5)
def plot_empirical_selected(key,l,d,all_metrics_df):
    for i in key:
        for_one_cell = all_metrics_df.loc[(all_metrics_df.seq_num == i)] # get all cycling data for 1 cell
        if len(for_one_cell) > 3:  # not select cells have too less data point
            y = for_one_cell['diag_discharge_capacity_rpt_0.2C']
            #x = for_one_cell['cycle_index']
            x = for_one_cell['equivalent_full_cycles']
            #y1 =  y / float(y.head(1)) 
            plt.plot(x,y)
            plt.xlabel("Equivalent Full Cycle")
            plt.ylabel("Capacity")
            plt.title("Capacity degradation",fontsize=16)
            plt.xlim(0,l)
            plt.ylim(d,5)
#predicted gamma
def plot_curve_pred(y,key,l,w,all_metrics_df,objective):
    '''
    y_with_key -> y??
    '''
    itera,num_var = y.shape
    num_var = num_var-1
    for i in key:
        cell_num = i
        for_one_cell = all_metrics_df.loc[(all_metrics_df.seq_num == cell_num)]

        #extract the cycle number
        x_range = np.floor(for_one_cell['equivalent_full_cycles'].max())
        x = np.arange(x_range)

        #predicted capacity
        selected = y[:,-1] == cell_num
        #test
        if num_var == 3:
            a,b,c = y[selected,0:num_var][0]
            acapacity_test_1 = objective(x,a,b,c)
        elif num_var == 4:
            a,b,c, d = y[selected,0:num_var][0]
            acapacity_test_1 = objective(x,a,b,c,d)
        elif num_var == 5:
            a,b,c, d , e = y[selected,0:num_var][0]
            acapacity_test_1 = objective(x,a,b,c,d,e)
        plt.plot(x,acapacity_test_1,label = '{}'.format(i))
    plt.xlabel("Equivalent Full Cycle")
    plt.ylabel("Capacity")
    plt.title("Capacity degradation",fontsize=16)
    if len(key)<2:
        plt.legend()
    plt.xlim(0,l)
    plt.ylim(w,5)

def correlation_heatmap(X_Y_merged,testParamDf):
    X_column_list = list(testParamDf.columns)[3:11]
    Y_column_list = ['cur_param_1','cur_param_2','cur_param_3']	
    Bayesian_test = X_Y_merged[['capacity_initial']+['resistance_initial']+X_column_list+Y_column_list]
    corrMatrix = Bayesian_test.corr()
    plt.figure(figsize=(16,6))
    sn.heatmap(corrMatrix, vmin=-1, vmax=1, annot=True,cmap='BrBG')
    plt.show()