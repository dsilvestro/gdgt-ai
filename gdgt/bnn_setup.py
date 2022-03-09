import numpy as np
import csv, os, sys
import pandas as pd
np.set_printoptions(suppress=True, precision=3)
from .prep_data import *
from .plot_results import *

def run_one_fold_bnn(f,
                     l,
                     rescale_factor,
                     rescale_labels,
                     cv_fold,
                     n_layers=[20, 10],
                     use_bias=True,
                     testsize=0.1,
                     n_iteration=10000000,
                     sampling_f=1000,
                     n_post_samples=1000,
                     verbose=0,
                     wd="",
                     seed=1234):
    np.random.seed(seed)
    rescale_feat = f * rescale_factor
    rescale_lab = rescale_MAT_data(l, rescale=rescale_labels)
    dat = get_data(rescale_feat, rescale_lab, cv=cv_fold, testsize=testsize, seed=seed)
    if use_bias:
        use_bias_node = 3

    bnn_model = bn.npBNN(dat,
                         n_nodes=n_layers,
                         estimation_mode="regression",
                         use_bias_node=use_bias_node
                         )
    mcmc = bn.MCMC(bnn_model,
                   n_iteration=n_iteration,
                   sampling_f=sampling_f,
                   print_f=10000,
                   n_post_samples=n_post_samples,
                   adapt_fM=0.6,
                   adapt_f=0.3,
                   estimate_error=True)

    mcmc._accuracy_lab_f(mcmc._y, bnn_model._labels)
    # initialize output files
    model_name = "bnn_cv%s" % cv_fold

    logger = bn.postLogger(bnn_model, filename=model_name, log_all_weights=0, wdir=wd)

    # run MCMC
    bn.run_mcmc(bnn_model, mcmc, logger)
    return logger._pklfile

def launch_run(arg):
    [f, l, rescale_factor, rescale_labels, cv, n_layers, cv_folds,
     use_bias, n_iteration, sampling_f, n_post_samples, verbose, out_dir] = arg
    p = run_one_fold_bnn(f, l,
                         rescale_factor=rescale_factor,
                         rescale_labels=rescale_labels,
                         cv_fold=cv,
                         n_layers=n_layers,
                         testsize=1 / cv_folds,
                         use_bias=use_bias,
                         n_iteration=n_iteration,
                         sampling_f=sampling_f,
                         n_post_samples=n_post_samples,
                         verbose=verbose,
                         wd=out_dir)
    return p

def run_cv_bnn(f,
               l,
               rescale_factor,
               rescale_labels,
               n_layers = [12,8,4],
               use_bias=True,
               cv_folds = 5,
               n_iteration=10000000,
               sampling_f=1000,
               n_post_samples=1000,
               verbose=1,
               wd="",
               parallel_run=True):
    out_dir = os.path.join(wd, 'bnn_logs')
    try:
        os.makedirs(out_dir)
    except:
        pass


    'parallel running of all 5 folds'
    list_args = []
    for cv in range(cv_folds):
        arg = [f, l, rescale_factor, rescale_labels, cv, n_layers, cv_folds,
             use_bias, n_iteration, sampling_f, n_post_samples, verbose, out_dir]
        list_args.append(arg)

    if parallel_run:
        import multiprocessing
        pool = multiprocessing.Pool(len(list_args))
        pkl_files = pool.map(launch_run, list_args)
        pool.close()
    else:
        pkl_files = []
        for arg in list_args:
            #if verbose:
            print("Running CV fold", cv)
            pkl_files.append(launch_run(arg))

    print("Trained models saved in:", out_dir)

    return pkl_files



def get_posterior_predictions(pkl_file, unlabeled_data, mat_model,save_predictions=True):
    bnn_obj, mcmc_obj, logger_obj = bn.load_obj(pkl_file)
    post_samples = logger_obj._post_weight_samples
    # reset data
    rescale_feat = unlabeled_data._features * mat_model._rescale_factor
    dat = get_data(rescale_feat, testsize=0)
    dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])
    dict_data = dictfilt(dat, ('data', 'labels', 'test_data', 'test_labels'))
    bnn_obj.update_data(dict_data)

    # load posterior weights
    post_weights = [post_samples[i]['weights'] for i in range(len(post_samples))]
    post_alphas = [post_samples[i]['alphas'] for i in range(len(post_samples))]
    if 'error_prm' in post_samples[0]:
        post_error = [post_samples[i]['error_prm'] for i in range(len(post_samples))]
    else:
        post_error = []
    actFun = bnn_obj._act_fun
    output_act_fun = bnn_obj._output_act_fun

    post_est = []
    post_est_test = []
    for i in range(len(post_weights)):
        actFun_i = actFun
        actFun_i.reset_prm(post_alphas[i])
        pred = bn.RunPredict(bnn_obj._data, post_weights[i], actFun=actFun_i, output_act_fun=output_act_fun)
        post_est.append(pred)

    post_est = np.array(post_est)
    prm_mean = np.mean(post_est,axis=0)[:,:]
    post_est_test = np.array(post_est_test)
    prm_mean_test = None

    if save_predictions:
        p = pd.DataFrame(prm_mean[:,0], columns=['predicted_MAT'])
        res_tbl = pd.concat([unlabeled_data._site, p], axis=1)
        outfile = os.path.basename(unlabeled_data._input_file) + "_BNNpredictions.txt"
        res_tbl.to_csv(os.path.join(unlabeled_data._wd, outfile), sep="\t", index=False)
        print("\nPredictions saved in:", os.path.join(unlabeled_data._wd, outfile))

    if 'error_prm' in post_samples[0]:
        # print("1-estimated error")
        dat_res = np.array([post_est[:, :, 0],
                            np.ones(post_est[:, :, 0].shape) * np.array(post_error)])
        mu = dat_res[0].T
        sd = dat_res[1].T
        pred_mean = np.random.normal(np.mean(mu, 1), np.mean(sd, 1), sd.shape[::-1])
        pred_mean = pred_mean.T
    else:
        pred_mean = None

    return {'point_estimates': prm_mean[:,0],
            'post_est': post_est, # samples from output layer
            'prm_mean_test': prm_mean_test,
            'post_est_test': post_est_test,
            'error_prm': post_error,
            'post_samples': pred_mean # samples based on posterior mu/sd
        }
