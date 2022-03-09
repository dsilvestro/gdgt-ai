import numpy as np
import csv, os
import pandas as pd
np.set_printoptions(suppress=True, precision=3)
from . import np_bnn as bn

def rescale_MAT_data(mat,reverse=False,rescale=True,clim_variable='MAT'):
    if rescale:
        if reverse:
            if clim_variable == 'MAT':
                return mat * 20 + 10
            else:
                return np.exp(mat)*1000
                # return mat * 1000
        else:
            if clim_variable == 'MAT':
                return (mat-10)/20
            else:
                return np.log(mat/1000)
                # return mat/1000
    else:
        return mat

def rescale_features(brGDGTdata, normalize=False):
    # calc fractional abundance
    a = np.array(brGDGTdata)
    b = np.sum(np.array(brGDGTdata), 1)
    brGDGTdata_FA = (a.T/b).T
    brGDGTdata_FAdf = pd.DataFrame(brGDGTdata_FA, columns=brGDGTdata.columns)
    if not normalize:
        rescale_factor = np.ones(brGDGTdata_FA.shape[1])
    else:
        rescale_factor = 1 / np.max(brGDGTdata_FA, 0) 
    return brGDGTdata_FAdf, rescale_factor
        


def get_data(f, l=None, cv=-1, testsize=0.1, seed=1234):
    dat = bn.get_data(f,
                      l,
                      seed=seed,
                      testsize=testsize, # 20% test set
                      all_class_in_testset=0,
                      cv=cv, # cross validation (1st batch; set to 1,2,... to run on subsequent batches)
                      instance_id=0,
                      header=1, # input data has a header
                      from_file=False,
                      randomize_order=True,
                      label_mode="regression")

    if l is not None:
        dat['labels'] = dat['labels'].reshape(len(dat['labels']),1).astype(float)
        if testsize:
            dat['test_labels'] = dat['test_labels'].reshape(len(dat['test_labels']),1).astype(float)
    return dat

