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

    argu_all = np.arange(1,3)

    loss_fun_0 = 'mse'
    loss_fun_1 = 'mae'
    loss_fun_all = [loss_fun_0,loss_fun_1]

    class_weight_all = []
    for i in range(1,10):
        for k in range(1,5,2):
            if i <= 2:
                class_weight_all.append({0: i,
                                    1: k,
                                    2: 1})
            elif i%2==0:
                class_weight_all.append({0: i,
                                    1: k,
                                    2: 1})
    nl_all = []
    for i in range(10,101,30):
        for j in range(i+30,101,30):
            nl_all.append([j,i])
            for k in range(j+30,101,30):
                nl_all.append([k,j,i])
                for z in range(k+30,101,30):
                    nl_all.append([z,k,j,i])
    opt_all = ['Adam', 'RMSprop']
    epochs_all = [150,300]
    rows = len(argu_all)*len(loss_fun_all)*len(class_weight_all)*len(nl_all) *len(opt_all)*len(epochs_all)
    result = pd.DataFrame(np.zeros((rows,9)),\
        columns=['argu','loss_fun', 'class_weight', 'nl', 'opt', 'epochs','Rsquare_median','R_greater_80','R_greater_85'])
        
    index = 0  
    pd.options.mode.chained_assignment = None  # default='warn'
    for i0 in [1]:
        model_raw_copy1 = copy.deepcopy(model_raw)
        model_raw_copy1.input_argumentation_update(i0)
        model_raw_copy2 = copy.deepcopy(model_raw_copy1)
        for i1 in ['mae']:
            for i2 in class_weight_all:
                for i3 in nl_all:
                    for i4 in opt_all:
                        for i5 in epochs_all:
                            model_raw_copy = copy.deepcopy(model_raw_copy2)
                            result['argu'][index] = i0
                            result['loss_fun'][index] = i1
                            result['class_weight'][index] = "{}".format(i2)
                            result['nl'][index] = "{}".format(i3)
                            result['opt'][index] = i4
                            result['epochs'][index] = i5
                            Rsquare = model_raw_copy.model_construction_and_training('zscore',all_metrics_df,i1,i2,i3,i4,i5,0)
                            result['Rsquare_median'][index] = np.median(Rsquare)
                            result['R_greater_80'][index] = np.mean(np.array(Rsquare) >= 0.80)
                            result['R_greater_85'][index] = np.mean(np.array(Rsquare) >= 0.85)
                            #model_raw_copy.model.save('models/model_{}'.format(index))
                            print (index,i0,i1,i2,i3,i4,i5,result['Rsquare_median'][index],result['R_greater_80'][index],result['R_greater_85'][index])
                            index+=1
            result.to_csv('result1_mae.csv')
    result.to_csv('result1_mae.csv')

    return result
    
if __name__ == '__main__':
    main()
