from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def norm (x,x_train,type):
    if type == 'range':
        return (x-x_train.min(axis=0))/(x_train.max(axis=0)-x_train.min(axis=0))
    if type == 'zscore':
        scaler = StandardScaler()
        scaler.fit(x_train)
        return scaler.transform(x)

def recover(x,x_train,type):
    if type == 'range':
        return x*(x_train.max(axis=0)-x_train.min(axis=0))+x_train.min(axis=0)
    if type == 'zscore':
        scaler = StandardScaler()
        scaler.fit(x_train)
        return scaler.inverse_transform(x)
def data_split(X,y_with_key,type):
    i = y_with_key.shape[1]
    X_copy = X.copy()
    y_with_key_copy = y_with_key.copy()
    X_train, X_test, y_train_coef, y_test_coef = train_test_split(X_copy, y_with_key_copy, test_size=0.15, random_state=42)
    y_train = y_train_coef[:,0:i-1]
    y_test = y_test_coef[:,0:i-1]

    y_train_norm = norm(y_train,y_train,type)
    y_test_norm =  norm(y_test,y_train,type)
    
    return X_train,X_test, y_train_coef,y_test_coef,y_train,y_test,y_train_norm,y_test_norm

def sequential_model(loss_method,X_train,y_train,nl,opt):
    in_dim= X_train.shape[1]
    out_dim= y_train.shape[1]
    model0 = Sequential()
    model0.add(Dense(nl[0], input_dim=in_dim, activation='relu'))
    if len(nl) == 2:
        model0.add(Dense(nl[1], activation='relu'))
    if len(nl) == 3:
        model0.add(Dense(nl[1], activation='relu'))
        model0.add(Dense(nl[2], activation='relu'))
    if len(nl) == 4:
        model0.add(Dense(nl[1], activation='relu'))
        model0.add(Dense(nl[2], activation='relu'))
        model0.add(Dense(nl[3], activation='relu'))
    model0.add(Dense(out_dim))
    model0.compile(loss=loss_method,
                    optimizer=opt,
                    metrics = ['accuracy'])
    return model0



def error_after_plot(y,yt,key,all_metrics_df,objective):
    itera,num_var = y.shape
    num_var = num_var-1
    R_sqaure = []
    for i in key:
        cell_num = i
        for_one_cell = all_metrics_df.loc[(all_metrics_df.seq_num == cell_num)]
        #extract the cycle number
        x_range = np.floor(for_one_cell['equivalent_full_cycles'].max())
        x = np.arange(x_range)
        selected = y[:,-1] == cell_num
        if num_var == 3:
            a,b,c = y[selected,0:num_var][0]
            at,bt,ct = yt[selected,0:num_var][0]
            acapacity_test_1 = objective(x,a,b,c)
            acapacity_test_t = objective(x,at,bt,ct)
        RSS = (np.sum(np.square(acapacity_test_t - acapacity_test_1)))
        TSS = (np.sum(np.square(acapacity_test_t-np.mean(acapacity_test_t))))
        R_sqaure.append(1-RSS/TSS)
    return R_sqaure



def pred_error(y,key,all_metrics_df,objective):
    rms={}
    mae={}
    num_var = y.shape[1]-1
    for i in key:
        cell_num = i
        for_one_cell = all_metrics_df.loc[(all_metrics_df.seq_num == cell_num)]
        capacity_exp = for_one_cell['diag_discharge_capacity_rpt_0.2C']
        #x = for_one_cell['cycle_index']
        cycle_number = for_one_cell['equivalent_full_cycles']
        #predicted capacity
        #predicted capacity
        selected = y[:,num_var] == cell_num
        if num_var == 3:
            a,b,c = y[selected,0:num_var][0]
            acapacity_test = objective(cycle_number,a,b,c)
        elif num_var == 4:
            a,b,c, d = y[selected,0:num_var][0]
            acapacity_test = objective(cycle_number,a,b,c,d)
        elif num_var == 5:
            a,b,c, d , e = y[selected,0:num_var][0]
            acapacity_test = objective(cycle_number,a,b,c,d,e)
        #selected = y[:,3] == cell_num
        #a_t1,b_t1,c_t1 = y[selected,0:num_var][0]
        #acapacity_test = objective(cycle_number,a_t1,b_t1,c_t1)

        # calculate errors
        rms[i] = np.sqrt(np.sum(np.square((acapacity_test - capacity_exp))) / len(capacity_exp)) * len(capacity_exp) / np.sum(capacity_exp)
        mae[i] = np.sum(np.abs(acapacity_test - capacity_exp)) / len(capacity_exp) 
        return rms, mae