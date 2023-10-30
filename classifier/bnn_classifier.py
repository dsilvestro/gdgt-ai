import numpy as np
np.set_printoptions(suppress=True, precision=3)
import sys
sys.path.insert(0, "/Users/dsilvestro/Software/npBNN")
# sys.path.insert(0, "/home/silvestr/Documents/npBNN")
import np_bnn as bn
import pandas as pd

from datetime import datetime
import scipy.stats
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

data = "/Users/dsilvestro/Software/gdgt-ai/R1/classifier/sedimentclass"

#--- TMP SETTINGS
cv_fold = 0      
rseed = 123
#---

f= "%s_features.txt" % data
l= "%s_labels.txt"   % data

fpd = pd.read_csv(f, delimiter="\t")
lpd = pd.read_csv(l, delimiter="\t")

dat = bn.get_data(fpd,lpd,
                  seed=rseed,
                  testsize=0.1, # 10% test set
                  all_class_in_testset=1,
                  header=1, # input data has a header
                  cv=cv_fold,
                  instance_id=1,
                  from_file=False) # input data includes names of instances



# set up model architecture and priors
n_nodes_list = [20,5] 
activation_function = bn.ActFun(fun="tanh")
use_bias_node = -1 

# set up the BNN model
bnn = bn.npBNN(dat,
               actFun=activation_function,
               use_bias_node=use_bias_node,
               use_class_weights=0,
               n_nodes = n_nodes_list,
               seed=rseed)


# set up the MCMC environment
mcmc = bn.MCMC(bnn,
               n_iteration=250000,
               sampling_f=1000,
               print_f=10000,
               n_post_samples=250,
               adapt_f=0.3,
               adapt_fM=0.6)



# initialize output files
output_name = "%s_cv%s" % (data, cv_fold)
logger = bn.postLogger(bnn, filename=output_name, log_all_weights=0, wdir="/home/silvestr/Documents/npBNN/pollination")

bn.run_mcmc(bnn, mcmc, logger)

# load results
pkl_file = "/Users/dsilvestro/Software/gdgt-ai/R1/classifier/sedimentclass_cv0_p1_h0_l20_5_s1_binf_123.pkl"

# predict
post_pr_test = bn.predictBNN(dat['test_data'],
                              pickle_file=pkl_file,
                              test_labels=dat['test_labels'],
                              instance_id=dat['id_test_data'],
                              fname=dat['file_name'],
                              post_summary_mode=1,
                              threshold=0.95)

# CALC TRADEOFFS
res = post_pr_test['post_prob_predictions']
labels=dat['test_labels']
all_y = all_y + list(res)
all_lab = all_lab + list(labels)


ppt = bn.get_posterior_threshold(pkl_file,
                                 target_acc=0.999,
                                 post_summary_mode=1,
                                 output_file=(pkl_file+"_thr.txt"))
