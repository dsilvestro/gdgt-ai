import os
import pandas as pd
import numpy as np
from .prep_data import *
from .nn_tf_setup import *
from .bnn_setup import *
from . import np_bnn as bn

class GDGTdata():
    def __init__(self,
                 infile,
                 brGDGDT_columnID='brGDGT', # keyword identifying brGDGT columns in the table
                 CLIM_columnID=None, # set to None is unlabeled data (for prediction)
                 normalize_features=True,
                 site_columnID='Site', # set to None if no site column is provided
                 clim_variable='MAT',
                 wd=''):
        self._input_file = infile
        self._normalize_features = normalize_features
        self._clim_variable = clim_variable
        if wd == '':
            self._wd = os.path.dirname(os.path.abspath(infile))
        else:
            self._wd = wd
        
        tbl = pd.read_csv(infile, sep=',')
        tbl = tbl.replace(' ', '_', regex=True)
        tbl.columns = tbl.columns.str.replace(' ', '_')

        # predictors
        brGDGTdata = tbl.filter(like=brGDGDT_columnID)
        self._features, self._rescale_factor = rescale_features(brGDGTdata, normalize=self._normalize_features)
        
        # labels
        if CLIM_columnID is not None:
            self._lCLIM = tbl[CLIM_columnID]
        else:
            self._lCLIM = None
            self._rescale_factor = None # for predict data not rescale factor is available


        if site_columnID is not None:
            self._site = tbl[site_columnID]
        else:
            self._site = pd.DataFrame(["site_%s" % i for i in range(tbl.shape[0])], columns="Site")

    def save_rescaled_features(self,
                               feature_file="brGDGDT_data.txt"):
        feature_tbl = pd.concat([self._site, self._features], axis=1)
        feature_tbl.to_csv(os.path.join(self._wd, feature_file), sep="\t", index=False)
        print("File saved to: \n", os.path.join(self._wd, feature_file))

    def save_labels(self,
                    mat_file="MAT_labels.txt"):
        if self._lCLIM is not None:
            label_tbl = pd.concat([self._site, self._features], axis=1)
            label_tbl.to_csv(os.path.join(self._wd, mat_file), sep="\t", index=False)
            print("File saved to: \n", os.path.join(self._wd, mat_file))

    def save_obj(self,
                 filename="brgdgt_data.pkl"):
        with open(os.path.join(self._wd, filename), 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


def load_model_settings(pkl_file):
    with open(pkl_file, 'rb') as f:
        model = pickle.load(f)
    return model


# Model object
class NNmodel():
    def __init__(self,
                 training_data,
                 wd=""):

        if training_data is None:
            try:
                with open(os.path.join(wd, "brgdgt_data.pkl"), 'rb') as f:
                    training_data = pickle.load(f)
            except:
                import pickle5
                with open(os.path.join(wd, "brgdgt_data.pkl"), 'rb') as f:
                    training_data = pickle5.load(f)
            training_data._wd = wd
            print("Using settings based on 'data/BNN_South_America_GDGT.csv' dataset")
        self._training_data = training_data
        self._rescale_factor = training_data._rescale_factor
        self._clim_variable = training_data._clim_variable
        self._wd = training_data._wd
        self._model_test_res = None
        self._nn_train_res = None


    ## functions
    def train_nn(self,
                 plot_predictions=True,
                 save_model=True,
                 epochs=100,
                 nodes=[12]):
        if self._model_test_res is not None:
            r = self._model_test_res['best_model_settings']
            nodes = r['nodes']
            epochs = r['epochs']

        res = train_nn_model(self._training_data._features,
                             self._training_data._lCLIM,
                             self._rescale_factor,
                             rescale_labels=True,
                             clim_variable=self._clim_variable,
                             n_epochs=epochs,
                             n_layers=nodes)

        if plot_predictions:
            plot_nn(res,
                    show=False,
                    filename=os.path.join(self._wd, res['model_name'] + ".pdf"))

        if save_model:
            model_dir = os.path.join(self._wd, res['model_name'])
            try:
                os.makedirs(model_dir)
            except:
                pass
            res['trained_model'].save(model_dir)

        self._nn_train_res = res

    def predict_nn(self,
                   unlabeled_data,
                   model_dir=None,
                   save_predictions=True):
        if self._nn_train_res is not None and model_dir is None:
            trained_model = self._nn_train_res['trained_model']
        else:
            trained_model = tf.keras.models.load_model(model_dir)
        'unlabeled_data (class: brGDGTdata)'  # used for prediction
        p = run_predict(unlabeled_data._features,
                        trained_model,
                        self._rescale_factor,
                        clim_variable=self._clim_variable,
                        rescale_labels=True)
        p = pd.DataFrame(p, columns=['predicted_MAT'])
        if save_predictions:
            res_tbl = pd.concat([unlabeled_data._site, p], axis=1)
            outfile = os.path.basename(unlabeled_data._input_file)+"_NN_predictions.txt"
            res_tbl.to_csv(os.path.join(unlabeled_data._wd, outfile), sep="\t", index=False)
            print("\nPredictions saved in:",os.path.join(unlabeled_data._wd, outfile))
        return p


    def model_testing_nn(self,
                         model_list=None,
                         max_epochs=1000,
                         cv_folds=5,
                         verbose=0,
                         plot_predictions=True,
                         save_summary_stats=True):

        if model_list is None:
            model_list = [
                [12],  # 0
                [12, 8],  # 1
                [12, 8, 4],  # 2
                [20, 12, 8, 4],  # 3
                # [20, 12, 8, 8, 4],
                [40, 20, 10, 5, 2]  # 5
            ]

        all_res = []
        for i in range(len(model_list)):
            model_name = "cv_" + "_".join(map(str, model_list[i]))
            cv_res, x_test, y_test = run_model_cv(self._training_data._features,
                                                  self._training_data._lCLIM,
                                                  self._rescale_factor,
                                                  rescale_labels=True,
                                                  clim_variable=self._clim_variable,
                                                  n_layers=model_list[i],
                                                  act_f=['relu', None],
                                                  cv_folds=cv_folds,
                                                  max_epochs=max_epochs,
                                                  verbose=verbose)
            summary = np.mean(cv_res, 0)
            print("Summary model: %s" % model_name)
            ep, se1, se2 = int(summary[0]), np.round(summary[1], 2), np.round(summary[2], 2)
            print("Epochs: %s RMSE training: %s RMSE test: %s " % (ep, se1, se2))
            full_res = {'model_name': model_name, 'cv_res': cv_res,
                        'true_labels': x_test, 'predicted_labels': y_test}
            if plot_predictions:
                plot_nn(full_res,
                        clim_variable=self._clim_variable,
                        reference_mat = 'true_labels',
                        show_rmse=True,
                        filename=os.path.join(self._wd,model_name + ".pdf"))
            all_res.append(full_res)



        # save summary stats of models
        if save_summary_stats:
            res_tbl = all_res[0]['cv_res'] + 0
            entry_names = ["_".join(map(str, model_list[0])) + "_cv%s" % j for j in range(cv_folds)]
            for i in range(1, len(model_list)):
                entry_names = entry_names + ["_".join(map(str, model_list[i])) + "_cv%s" % j for j in range(cv_folds)]
                res_tbl = np.concatenate((res_tbl, all_res[i]['cv_res']))
            res_tbl_df = pd.DataFrame(res_tbl, columns=["epochs", "rmse_train", "rmse_test"], index=entry_names)
            res_tbl_df.to_csv(os.path.join(self._wd, "summary_TF_nn_training.txt"), sep="\t")

        # choose best model
        mean_res = np.array([np.mean(all_res[i]['cv_res'], 0) for i in range(len(model_list))])
        best_model = np.argmin(mean_res[:, 2])
        best_model_settings = {'epochs': int(mean_res[best_model][0]),
                               'nodes': model_list[best_model]
                               }
        self._model_test_res = {'best_model_settings': best_model_settings,
                                'all_results': all_res}

    def save_model_settings(self, filename):
        'save object to a pkl file'
        with open(os.path.join(self._wd, filename), 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


    def cross_validation_bnn(self,
                             n_layers=[12, 8, 4],
                             rescale_labels=False,
                             cv_folds=5,
                             n_iteration=10000000,
                             sampling_f=1000,
                             n_post_samples=1000,
                             verbose=1,
                             parallel_run=True,
                             plot_predictions=False):

        pkl_files = run_cv_bnn(self._training_data._features,
                               self._training_data._lCLIM,
                               self._rescale_factor,
                               rescale_labels=rescale_labels,
                               n_layers=n_layers,
                               cv_folds=cv_folds,
                               n_iteration=n_iteration,
                               sampling_f=sampling_f,
                               n_post_samples=n_post_samples,
                               verbose=verbose,
                               wd=self._wd,
                               parallel_run=parallel_run)

        dat_res = []
        lab = []
        mu = []
        sd = []

        model_name = "bnn_cv_" + "_".join(map(str, n_layers))
        

        # get test data IDs
        f = pd.concat([self._training_data._site, self._training_data._features], axis=1)
        l = pd.concat([self._training_data._site, self._training_data._lCLIM], axis=1)

        test_IDs = []
        for i in range(cv_folds):
            dat = bn.get_data(f,
                              l,
                              cv=i,
                              testsize=0.2,
                              seed=1234,
                              instance_id=1,
                              header=1,  # input data has a header
                              from_file=False,
                              randomize_order=True,
                              label_mode="regression")
            test_IDs = test_IDs + list(dat['id_test_data'])
 
        for cv in range(cv_folds):
            # load predictions
            with open(pkl_files[cv], 'rb') as f:
                pkl = pickle.load(f)

            x_test = pkl[0]._test_labels

            res = bn.get_posterior_est(pkl_files[cv])

            y_test = rescale_MAT_data(res['prm_mean_test'][:, 0], reverse=True, rescale=rescale_labels)
            RMSE_test = np.sqrt(np.mean((x_test - y_test) ** 2))

            try:
                dat_res.append(
                    [x_test.flatten(), res['post_est_test'][:, :, 0], res['post_est_test'][:, :, 1]])
            except:
                "1-estimated error"
                dat_res.append([x_test, res['post_est_test'][:, :, 0],
                                np.ones(res['post_est_test'][:, :, 0].shape) * np.array(res['error_prm'])])

            # print("Summary model: %s" % model_name)
            # print("RMSE test: %s " % RMSE_test)

            lab = lab + list(dat_res[cv][0])
            mu.append(dat_res[cv][1].T)
            sd.append(dat_res[cv][2].T)

        lab = np.array(lab)
        mu = np.array(mu).reshape((len(lab), dat_res[0][1].shape[0]))
        sd = np.array(sd).reshape((len(lab), dat_res[0][1].shape[0]))

        lab = rescale_MAT_data(lab, reverse=True, rescale=rescale_labels)

        z_test = np.random.normal(mu, sd, sd.shape)
        z_test = rescale_MAT_data(z_test, reverse=True, rescale=rescale_labels)
        ci_size = np.array([bn.calcHPD(z_test[i,:], 0.95) for i in range(z_test.shape[0])])
        res = {'true_labels': lab, 'sampled_labels': z_test,
               'model_name': model_name, 'predicted_labels': mu,
               'CI': ci_size, 'plot_title': "BNN - CV test sets "}
        model_name = res['model_name']

        if plot_predictions:
            plot_bnn_training(res,
                     show=False,
                     filename=os.path.join(self._wd, model_name + ".pdf"),
                     rescale_labels=rescale_labels,
                     title="BNN - CV test sets ")

            # plot_bnn2(res,
            #          filename=os.path.join(self._wd, model_name + "PLOT2.pdf"),
            #          )
        
        
        # save CV test predictions
        outres_file = os.path.join(self._wd, model_name + "_test_IDS_and_predictions.txt")
        pred = pd.DataFrame(np.array([test_IDs, 
                                     res['true_labels'][:,0], 
                                     np.mean(res['predicted_labels'],1), 
                                     res['CI'][:,0],
                                     res['CI'][:,1]
                                 ]).T)
        pred.to_csv(outres_file, sep="\t", index=False)
        print("Prediciton of the CV test sets saved as:\n", outres_file)
        
        

        return res

    def train_bnn(self,
                  n_layers=[12, 8, 4],
                  n_iteration=10000000,
                  sampling_f=1000,
                  n_post_samples=1000,
                  testsize=0.1,
                  verbose=1,
                  use_bias=True,
                  rescale_labels=False
                  ):

        pkl_file = run_one_fold_bnn(self._training_data._features, self._training_data._lCLIM,
                         self._rescale_factor,
                         rescale_labels=rescale_labels,
                         cv_fold=-1,
                         n_layers=n_layers,
                         use_bias=use_bias,
                         testsize=testsize,
                         n_iteration=n_iteration,
                         sampling_f=sampling_f,
                         n_post_samples=n_post_samples,
                         verbose=verbose,
                         wd=self._wd,
                         seed=1234)
        
        print("Trained model saved as: ", pkl_file)
        return pkl_file




def run_bnn_cv_predictions(pkl_files,
                           n_layers=[12, 8, 4],
                           rescale_labels=False,
                           cv_folds=5,
                           plot_predictions=False):
    dat_res = []
    lab = []
    mu = []
    sd = []

    model_name = "bnn_cv_" + "_".join(map(str, n_layers))
    for cv in range(cv_folds):
        with open(pkl_files[cv], 'rb') as f:
            pkl = pickle.load(f)

        x_test = pkl[0]._test_labels

        res = bn.get_posterior_est(pkl_files[cv])

        y_test = rescale_MAT_data(res['prm_mean_test'][:, 0], reverse=True, rescale=rescale_labels)
        RMSE_test = np.sqrt(np.mean((x_test - y_test) ** 2))

        try:
            dat_res.append(
                [x_test.flatten(), res['post_est_test'][:, :, 0], res['post_est_test'][:, :, 1]])
        except:
            "1-estimated error"
            dat_res.append([x_test, res['post_est_test'][:, :, 0],
                            np.ones(res['post_est_test'][:, :, 0].shape) * np.array(res['error_prm'])])

        # print("Summary model: %s" % model_name)
        # print("RMSE test: %s " % RMSE_test)

        lab = lab + list(dat_res[cv][0])
        mu.append(dat_res[cv][1].T)
        sd.append(dat_res[cv][2].T)

    lab = np.array(lab)
    mu = np.array(mu).reshape((len(lab), dat_res[0][1].shape[0]))
    sd = np.array(sd).reshape((len(lab), dat_res[0][1].shape[0]))

    lab = rescale_MAT_data(lab, reverse=True, rescale=rescale_labels)

    z_test = np.random.normal(mu, sd, sd.shape)
    z_test = rescale_MAT_data(z_test, reverse=True, rescale=rescale_labels)

    res = {'true_labels': lab, 'sampled_labels': z_test,
           'model_name': model_name, 'predicted_labels': mu}

    if plot_predictions:
        plot_bnn(res,
                 show=False,
                 filename=os.path.join(model_name + "_CV_testset.pdf"),
                 rescale_labels=rescale_labels)

    return res
















