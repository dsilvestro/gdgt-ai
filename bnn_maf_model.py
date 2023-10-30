import numpy as np
np.set_printoptions(suppress=True, precision=3)
import sys
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
# load GDGT-AI library

gdgt_ai_path = "specify path to the gdgt-ai folder"
sys.path.insert(0, gdgt_ai_path)
import gdgt

variable = "MAF"
wd = "training_data"
training_data_file = os.path.join(gdgt_ai_path, wd, 'soil_peat_features.txt')

tbl = pd.read_csv(training_data_file, sep='\t')

rs = RandomState(MT19937(SeedSequence(123)))
tbl_red = tbl.iloc[rs.choice(range(tbl.shape[0]), size=tbl.shape[0], replace=False)]
tbl_red.to_csv(training_data_file + "_rnd.txt",index=False, sep='\t')


training_data_file = training_data_file + "_rnd.txt" 
print(training_data_file)

#---- train/test BNN with cross validation ----#
# prep training data
training_data = gdgt.GDGTdata(training_data_file,
                              brGDGDT_columnID='fI', # keyword identifying brGDGT columns in the table
                              CLIM_columnID=variable, # set to None is unlabeled data (for prediction)
                              normalize_features=True,
                              sep='\t',
                              site_columnID="Sample"
                              )


# set up matnn model
model = gdgt.NNmodel(training_data)
# save model settings
model.save_model_settings(filename='%s_model_settings.pkl' % variable)

p = model.cross_validation_bnn(n_layers=[12,4],
                               n_iteration=250000000,
                               sampling_f=10000,
                               n_post_samples=1000,
                               parallel_run=False, # run on multiple CPUs if set True
                               plot_predictions=True,
                               rescale_labels=False)

