
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

def data_generation():
    '''
    Generate the coefficient to represent the trajectory curve
    '''
    PreDiag_parameters = 'Data/battery_cycling_data/PreDiag_parameters.csv'
    testParamDf = pd.read_csv(PreDiag_parameters)

    testParamDf['charging_protocol'] = testParamDf['charge_constant_current_1'].astype(str) + '_'+ testParamDf['charge_constant_current_2'].astype(str) + '_'+ testParamDf['charge_cutoff_voltage'].astype(str) +'_'+ testParamDf['charge_constant_voltage_time'].astype(str) 
    testParamDf['CC1_CC2'] = testParamDf['charge_constant_current_1'].astype(str) + '_'+ testParamDf['charge_constant_current_2'].astype(str)
    testParamDf['discharging_protocol'] = testParamDf['discharge_constant_current'].astype(str) +  '_'+ testParamDf['discharge_cutoff_voltage'].astype(str) 
    testParamDf['DOD'] = testParamDf['charge_cutoff_voltage'].astype(str) + '_' + testParamDf['discharge_cutoff_voltage'].astype(str)

    # testParamDf
    DODs = len(testParamDf['DOD'].unique())
    print(f'Unique DODs {DODs}')
    discharge_protocols = len(testParamDf['discharging_protocol'].unique())
    print(f'Unique discharging_protocol {discharge_protocols}')
    charge_protocols = len(testParamDf['charging_protocol'].unique())
    print(f'Unique charging_protocol {charge_protocols}')

    print(len(testParamDf['discharge_cutoff_voltage'].unique()))
    print(len(testParamDf['CC1_CC2'].unique()))
    print(len(testParamDf['discharge_constant_current'].unique()))

    # (0) top left - RPT-calculated Capacities
    rpt_metrics_df_BVV_10_06_2021_TOTAL = 'Data/battery_cycling_data/rpt_metrics_df_BVV_10_06_2021_TOTAL.csv'

    RPT_metrics_df = pd.read_csv(rpt_metrics_df_BVV_10_06_2021_TOTAL).drop('Unnamed: 0',axis=1)
    RPT_metrics_df['cycle_index'] = RPT_metrics_df['diag_cycle_index_reset']+2

    # (1) top right - Resistances at 3 different SOCs
    ResistanceCombined = 'Data/battery_cycling_data/ResistanceCombined.csv'
    hppc_res_df = pd.read_csv(ResistanceCombined).set_index(['cycle_index'])
    hppc_res_df['r_tot_c_0'] = hppc_res_df['r_tot_c_0']*1000
    hppc_res_df['r_tot_c_5'] = hppc_res_df['r_tot_c_5']*1000
    hppc_res_df['r_tot_c_8'] = hppc_res_df['r_tot_c_8']*1000

    all_metrics_df = pd.merge(hppc_res_df, RPT_metrics_df,  how='left', left_on=['seq_num','cycle_index'], right_on = ['seq_num','cycle_index'])
    return all_metrics_df,testParamDf

def objective(x, a, b, c):
    '''
    Assumped model to represent trajectory curve
    '''
    return a * x * x + b * x + c

def raw_input(all_metrics_df,testParamDf):
    '''
    Generate X and y raw input
    '''
    import math
    from numpy import arange
    from scipy.optimize import curve_fit
    Y={}

    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler


    # for loop to fit capacity curve for all cells
    for i in range(328):
        for_one_cell = all_metrics_df.loc[(all_metrics_df.seq_num == list(all_metrics_df.seq_num.unique())[i])] # get all cycling data for 1 cell
        if len(for_one_cell) > 3:  # not select cells have too less data point
            y = for_one_cell['diag_discharge_capacity_rpt_0.2C']
            #x = for_one_cell['cycle_index']
            x = for_one_cell['equivalent_full_cycles']
            #y1 =  y / float(y.head(1)) 
            #X = flat(scaler_A.transform([[b] for b in x]))
            r_dis_oh = for_one_cell['r_d_o_0']
            r_dis_ct = for_one_cell['r_d_ct_0']
            r_dis_pul = for_one_cell['r_d_p_0']
            r_tot = r_dis_oh + r_dis_ct + r_dis_pul
            
            try:                           
                popt, _ = curve_fit(objective, x, y, bounds=([-1e-4,-0.002,4],[5e-5,0.02,5]))  # fit curve
            except:
                continue   # if not fit, continue to next cell
            else:
                a, b, c= popt
                k = for_one_cell.cycle_index.keys()[0]   
                key = all_metrics_df.seq_num[k]       # find seq_num of this cell
                y_pre = np.array([objective(x2, a, b,c) for x2 in x])  #calculate y_pre based on fiiting function
                rms = np.sqrt(np.sum(np.square((y_pre - y))) / len(y)) * 100 * len(y) / np.sum(y)  # calculate %RMS
                if rms < 5:   # select fittings within 5% RMS
                    Y[key] = [popt,rms,float(y.head(1)),float(r_tot.iloc[1])]  #store fitting params, %rms, initial capacity and resistance
    # create pd for Y matrix
    seq_num = pd.Series(Y.keys())
    cur_param = pd.Series(Y.values())
    cur_param_1 = pd.Series([cur[0][0] for cur in cur_param])
    cur_param_2 = pd.Series([cur[0][1] for cur in cur_param])
    cur_param_3 = pd.Series([cur[0][2] for cur in cur_param])
    cur_err = pd.Series([cur[1] for cur in cur_param])
    capacity_initial = pd.Series([cur[2] for cur in cur_param])
    resistance_initial = pd.Series([cur[3] for cur in cur_param])
    Y_matrix_pre = pd.DataFrame({'seq_num': seq_num, 'cur_param_1': cur_param_1, 'cur_param_2': cur_param_2, 'cur_param_3': cur_param_3,
                                'cur_err':cur_err, 'capacity_initial':capacity_initial, 'resistance_initial':resistance_initial})
    #display(Y_matrix_pre)
    # prepare X matrix
    X_column_list = list(testParamDf.columns)[3:11]
    X_matrix = testParamDf[['seq_num'] + X_column_list]
    # merge X and Y matrix based on seq_num
    X_Y_merged = pd.merge(Y_matrix_pre, X_matrix,  how='left', left_on=['seq_num'], right_on = ['seq_num'])
    X_Y_merged = X_Y_merged.dropna()
    # normalize input X 
    scaler=preprocessing.StandardScaler().fit(X_Y_merged[['capacity_initial']+['resistance_initial']+X_column_list])
    X = scaler.transform(X_Y_merged[['capacity_initial']+['resistance_initial']+X_column_list])
    #display(X)
    y_matrix = [y[0] for y in Y.values()] 
    y = np.array(y_matrix)
    return X,y,Y,X_Y_merged