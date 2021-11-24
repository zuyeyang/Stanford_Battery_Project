from B_model import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
def main(plot=True):
    all_metrics_df,testParamDf = data_generation()
    test_coef = ["0"]
    func_coef = ["raw"]
    version = '0' #raw ones
    X,y,Y,X_Y_merged = eval(func_coef[0]+"_order_poly_fitting(all_metrics_df,testParamDf)")
    objective = eval('objective_'+version)    

    model_raw = battery_model(0,X,y,Y,X_Y_merged,objective)

    argu_all = np.arange(1,5)

    loss_fun_0 = 'mse'
    loss_fun_1 = 'mae'
    loss_fun_all = [loss_fun_0,loss_fun_1]

    class_weight0 = {0: 1.,
                1: 1.,
                2: 1.}
    class_weight1 = {0: 10000.,
                    1: 100.,
                    2: 1.}   
    class_weight2 = {0: 1000000.,
                    1: 1000.,
                    2: 1.}       
    class_weight3 = {0: 100000000.,
                    1: 10000.,
                    2: 1.}    
    class_weight_all = [class_weight0, 
                    class_weight1,
                    class_weight2,
                    class_weight3]

    nl_all = []
    for i in range(10,101,30):
        for j in range(i+20,101,30):
            nl_all.append([j,i])
            for k in range(j+20,101,30):
                nl_all.append([k,j,i])
                for z in range(k+20,101,30):
                    nl_all.append([z,k,j,i])
    opt_all = ['SGD', 'Adam', 'RMSprop']
    epochs_all = [100,200]
    result = pd.DataFrame(np.zeros((len(argu_all)*len(class_weight_all)*len(nl_all) *len(opt_all)*len(epochs_all),7)),\
        columns=['argu','loss_fun', 'class_weight', 'nl', 'opt', 'epochs','error'])
        
    index = 0  
    for i0 in argu_all:
        model_raw_copy = copy.deepcopy(model_raw)
        model_raw_copy.input_argumentation_update(i0)
        for i1 in loss_fun_all:
            for i2 in class_weight_all:
                for i3 in nl_all:
                    for i4 in opt_all:
                        for i5 in epochs_all:
                            result['argu'][index] = i0
                            result['loss_fun'][index] = i1
                            result['class_weight'][index] = "{}".format(i2)
                            result['nl'][index] = "{}".format(i3)
                            result['opt'][index] = i4
                            result['epochs'][index] = i5
                            result['error'][index] = model_raw_copy.model_construction_and_training(all_metrics_df,i1,i2,i3,i4,i5,0)
                            index+=1
                            print (i0,i1,i2,i3,i4,i5)
                            
        result.to_csv('result.csv')

        return result
    
if __name__ == '__main__':
    main()
