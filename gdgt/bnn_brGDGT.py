import numpy as np
import csv, os, sys
import pandas as pd
np.set_printoptions(suppress=True, precision=3)
import argparse
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf  # saves pdfs

# import prep_data
# import nn_tf_setup

# sys.path.insert(0, r'/Users/dsilvestro/Software/npBNN/')
import np_bnn as bn

wd = "/Users/dsilvestro/Software/experimental_code/geology_models/brGDGT/code_n_data"
# wd = "."

## GET DATA
f = os.path.join(wd, "features_FArescaled.txt")
l = os.path.join(wd, "labelMAT.txt") # "labelMATrescaled.txt"

# parse arguments
cmd_line = 1
if cmd_line:
    p = argparse.ArgumentParser()
    p.add_argument('-r', type=int, help='seed', default = 1234)
    p.add_argument('-w', type=int, help='', default = [12,8,2], nargs="+")
    p.add_argument('-cv', type=int, help='', default = 0)
    args = p.parse_args()
    rseed = args.r
    cross_validation_batch = args.cv
    n_nodes_list = args.w
else:
    rseed = 1234
    cross_validation_batch = 0
    n_nodes_list = [12,8,4]


np.random.seed(rseed)

dat = bn.get_data(f,
                  l,
                  seed=1234,
                  testsize=0.2, # 20% test set
                  all_class_in_testset=0,
                  cv=cross_validation_batch,
                  instance_id=1,
                  header=1, # input data has a header
                  from_file=True,
                  randomize_order=True,
                  label_mode="regression")

dat['labels'] = dat['labels'].reshape(len(dat['labels']),1).astype(float)
dat['test_labels'] = dat['test_labels'].reshape(len(dat['test_labels']),1).astype(float)

# set up the BNN model
bnn_model = bn.npBNN(dat,
                     n_nodes = n_nodes_list,
                     estimation_mode="regression",
                     use_bias_node=3
)

# set up the MCMC environment
mcmc = bn.MCMC(bnn_model,
               n_iteration=25000000,
               sampling_f=1000,
               print_f=10000,
               n_post_samples=1000,
               adapt_fM=0.6,
               adapt_f=0.3,
               estimate_error=True)



mcmc._accuracy_lab_f(mcmc._y, bnn_model._labels)
# initialize output files
model_name = "_cv%s" % cross_validation_batch
model_name = "_".join(map(str, n_nodes_list))

logger = bn.postLogger(bnn_model, filename="brgdgt" + model_name, log_all_weights=0)

# run MCMC
bn.run_mcmc(bnn_model, mcmc, logger)


