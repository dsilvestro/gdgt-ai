import numpy as np
import csv, os, sys
import pandas as pd
np.set_printoptions(suppress=True, precision=3)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf  # saves pdfs

from .prep_data import *
from  .nn_tf_setup import *

sys.path.insert(0, r'/Users/dsilvestro/Software/npBNN/')
import np_bnn as bn


wd = "/Users/dsilvestro/Software/experimental_code/geology_models/brGDGT/code_n_data"
dir_model = "/Users/dsilvestro/Documents/Projects/Ongoing/Jaramillo_brGDGT"
## GET DATA
plot = False
verbose = 0
f = os.path.join(wd, "features_FArescaled.txt")
l = os.path.join(wd, "labelMATrescaled.txt") 

## RUN NN with cross validation
def run_model_testing():
    model_list = [
        [12],          # 0
        [12,8],        # 1
        [12,8,4],      # 2
        [20,12,8,4],   # 3
        [20,12,8,8,4], # 4
        [40,20,10,5,2], # 5
        [200, 40, 20, 10, 5, 2]  # 5
    ]
    
    
    all_res = []
    for i in range(len(model_list)):
        cv_res,x_test,y_test = nn_tf_setup.run_model_cv(f,
                                                        l,
                                                        n_layers = model_list[i],
                                                        act_f = ['relu', None],
                                                        cv_folds = 5,
                                                        rescale_labels=True,
                                                        verbose=0)
        print("summary MODEL %s: \n" % i, np.mean(cv_res,0))
        all_res.append([cv_res,x_test,y_test])
    
    # save summary stats of models
    res_tbl = all_res[0][0] + 0
    entry_names = ["_".join(map(str, model_list[0])) + "_cv%s" % j for j in range(5)]
    for i in range(1,len(model_list)):
        entry_names = entry_names + ["_".join(map(str, model_list[i])) + "_cv%s" % j for j in range(5)]
        res_tbl = np.concatenate((res_tbl,all_res[i][0] ))
    res_tbl_df = pd.DataFrame(res_tbl, columns = ["epochs","rmse_train", "rmse_test"], index=entry_names)
    res_tbl_df.to_csv(os.path.join(wd, "results/summary_TF_nn_training.txt"), sep="\t")
    
    # choose best model
    mean_res = np.array([np.mean(all_res[i][0], 0) for i in range(len(model_list))])
    best_model = np.argmin(mean_res[:,2])
    [cv_res, x_test, y_test] = all_res[best_model]
    
    
    
    # train 'production' model
    n_layers = model_list[best_model]
    n_epochs = int(np.mean(cv_res[:,0]))
    m_name = "results/retrained" + "_".join(map(str, n_layers))
    _, x_prod, y_prod, m = nn_tf_setup.run_production_model(f,l,
                                         n_epochs=n_epochs,
                                         n_layers = n_layers,
                                         use_bias=True,
                                         act_fs = ['relu', None],
                                         verbose=0,
                                         plot=True)
    model_dir = os.path.join(dir_model,m_name)
    try: os.makedirs(model_dir)
    except: pass
    m.save(model_dir)
    
    # save labels and predictions
    res_tbl_df = pd.DataFrame(np.array([x_test, y_test]).T, columns = ["true_MAT", "predicted_MAT"])
    res_tbl_df.to_csv(os.path.join(wd, "results/test_sets_%s.txt" % "_".join(map(str, n_layers))), sep="\t",index=False)
    
    res_tbl_df = pd.DataFrame(np.array([x_prod, y_prod]).T, columns = ["true_MAT", "predicted_MAT"])
    res_tbl_df.to_csv(os.path.join(wd, "results/train_set_production_%s.txt" % "_".join(map(str, n_layers))), sep="\t",index=False)


# feature importance
def run_feature_importance():
    m = "/Users/dsilvestro/Documents/Projects/Ongoing/Jaramillo_brGDGT/retrained40_20_10_5_2"
    run_feature_importance(f,
                           l,
                           model_dir=m,
                           cv=-1,
                           testsize=0,
                           n_permutations=100,
                           verbose=1,
                           seed=1234)