import numpy as np
import csv, os, sys
import pandas as pd
np.set_printoptions(suppress=True, precision=3)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf  # saves pdfs
class MCDropout(tf.keras.layers.Layer):
    def __init__(self, rate):
        super(MCDropout, self).__init__()
        self.rate = rate

    def call(self, inputs):
        return tf.nn.dropout(inputs, rate=self.rate)
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable tf compilation warning
except:
    pass

from .prep_data import *
from .plot_results import *

sys.path.insert(0, r'/Users/dsilvestro/Software/npBNN/')
import np_bnn as bn

def build_regression_model(train_set, 
                           n_layers, 
                           act_f = 'relu',
                           dropout=None,
                           dropout_rate=0,
                           use_bias=True,
                           act_f_out=None):
    architecture = [tf.keras.layers.Flatten(input_shape=[train_set.shape[1]])]
    architecture.append(tf.keras.layers.Dense(n_layers[0],
                                  activation=act_f,
                                  use_bias=use_bias))
    for i in n_layers[1:]:
        architecture.append(tf.keras.layers.Dense(i, activation=act_f))

    if dropout:
        dropout_layers = [MCDropout(dropout_rate) for i in architecture[1:]]
        architecture = [architecture[0]] + [j for i in zip(architecture[1:],dropout_layers) for j in i]

    if act_f_out:
        architecture.append(tf.keras.layers.Dense(1, activation=act_f_out))    #sigmoid or tanh
    else:
        architecture.append(tf.keras.layers.Dense(1))
    model = tf.keras.Sequential(architecture)
    optimizer = "adam"       # "adam" or tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mae','mse'])
    return model
    

def run_one_fold(f,
                 l,
                 rescale_factor,
                 rescale_labels,
                 clim_variable,
                 cv_fold,
                 n_layers = [20, 10],
                 use_bias=True,
                 testsize=0.1,
                 patience=10,
                 act_fs = ['relu', None],
                 max_epochs = 1000,
                 plot=False, 
                 verbose=0,
                 seed=1234):
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rescale_feat = f * rescale_factor
    rescale_lab = rescale_MAT_data(l,rescale=rescale_labels, clim_variable=clim_variable)
    dat = get_data(rescale_feat, rescale_lab, cv=cv_fold, testsize=testsize, seed=seed)
    
    model = build_regression_model(dat['data'], 
                                   n_layers, 
                                   act_f = act_fs[0],
                                   dropout=None,
                                   dropout_rate=0,
                                   use_bias=use_bias,
                                   act_f_out=act_fs[1])
    if verbose:
        model.summary()

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mse', 
                                                  patience=patience, 
                                                  restore_best_weights=True)

    history = model.fit(dat['data'], dat['labels'],
                        epochs=max_epochs, 
                        validation_data=(dat['test_data'],dat['test_labels']),
                        verbose=0,
                        callbacks=[early_stop])
    
    stopping_point = np.argmin(history.history['val_mse'])+1
    if verbose:
        print('Best training epoch: ',stopping_point,flush=True)
    

    if plot:
        fig = plt.figure(figsize=(5, 5))
        plt.plot(history.history['mse'][100:])
        plt.plot(history.history['val_mse'][100:])
        fig.show()
        

    y_train = model.predict(dat['data'], verbose=verbose)
    y_test = model.predict(dat['test_data'], verbose=verbose)
    

    x_train = rescale_MAT_data(dat['labels'].flatten(), reverse=True,rescale=rescale_labels, clim_variable=clim_variable)
    y_train = rescale_MAT_data(y_train, reverse=True,rescale=rescale_labels, clim_variable=clim_variable).flatten()
    RMSE_train = np.sqrt(np.mean((x_train - y_train)**2))
    x_test= rescale_MAT_data(dat['test_labels'].flatten(), reverse=True,rescale=rescale_labels, clim_variable=clim_variable)
    y_test= rescale_MAT_data(y_test, reverse=True,rescale=rescale_labels, clim_variable=clim_variable).flatten()
    RMSE_test = np.sqrt(np.mean((x_test - y_test)**2))
    
    if plot:
        fig = plt.figure(figsize=(5, 5))
        g = sns.regplot(x=x_train, y=y_train, label="Training set (%s)" % np.round(RMSE_train,2))
        sns.regplot(x=x_test, y=y_test, label="Test set (%s)" % np.round(RMSE_test, 2))
        plt.axline((0, 0), (1, 1), linewidth=2, color='k')
        g.set(ylim=(-5, 30), xlim=(-5, 30))
        g.legend(loc=2)
        plt.xlabel('Observed means')
        plt.ylabel('Predicted means')        
        title = "NN regression"
        plt.title(title)   
        fig.show()
    
    if verbose:
        print("RMSE train", RMSE_train)
        print("RMSE test",  RMSE_test)
    res = stopping_point, RMSE_train, RMSE_test, history
    return res, x_test, y_test, model


def train_nn_model(f,
                   l,
                   rescale_factor,
                   rescale_labels,
                   clim_variable,
                   n_epochs,
                   n_layers,
                   use_bias=True,
                   act_fs = ['relu', None],
                   testsize=0,
                   plot=False,
                   verbose=0,
                   seed=1234):
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rescale_feat = f * rescale_factor
    rescale_lab = rescale_MAT_data(l,rescale=rescale_labels,clim_variable=clim_variable)
    dat = get_data(rescale_feat, rescale_lab, cv=-1, testsize=testsize, seed=seed)
    
    model = build_regression_model(dat['data'], 
                                   n_layers, 
                                   act_f = act_fs[0],
                                   dropout=None,
                                   dropout_rate=0,
                                   use_bias=use_bias,
                                   act_f_out=act_fs[1])
    if verbose:
        model.summary()

    model.fit(dat['data'], dat['labels'],
              epochs=n_epochs,
              validation_split=0,
              verbose=0)

    y_train = model.predict(dat['data'], verbose=verbose)
    # rescale labels
    x_train = rescale_MAT_data(dat['labels'].flatten(), reverse=True,rescale=rescale_labels,clim_variable=clim_variable)
    y_train = rescale_MAT_data(y_train, reverse=True,rescale=rescale_labels,clim_variable=clim_variable).flatten()
    RMSE_train = np.sqrt(np.mean((x_train - y_train)**2))
    
    if plot:
        fig = plt.figure(figsize=(5, 5))
        sns.regplot(x=(dat['labels'].flatten()), y=y_train)
        fig.show()


    print("RMSE train", RMSE_train)
    model_name = "nn_" + "_".join(map(str, n_layers))
    res = {'model_name': model_name,
           'true_labels': x_train, 'predicted_labels': y_train,
           'trained_model': model}
    return res


def run_model_cv(f,
                 l,
                 rescale_factor,
                 rescale_labels,
                 clim_variable,
                 n_layers = [12,8,4],
                 act_f = ['relu', None],
                 cv_folds = 5,
                 max_epochs=1000,
                 verbose=1):
    cv_res = []
    x_test = []
    y_test = []
    for cv in range(cv_folds):
        if verbose: 
            print("Running CV fold", cv)
        res, x, y, m = run_one_fold(f, l,
                                    rescale_factor=rescale_factor,
                                    rescale_labels=rescale_labels,
                                    clim_variable=clim_variable,
                                    cv_fold=cv,
                                    n_layers=n_layers,
                                    testsize=1 / cv_folds,
                                    use_bias=True,
                                    patience=50,
                                    act_fs=act_f,
                                    max_epochs = max_epochs,
                                    plot=False, 
                                    verbose=verbose
                                    )
                       
        cv_res.append(res[:-1])
        x_test = x_test + list(x)
        y_test = y_test + list(y)
    
    cv_res = np.array(cv_res)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return cv_res, x_test, y_test
    

def run_feature_importance(f,
                           l,
                           model_dir,
                           cv=-1,
                           testsize=0,
                           n_permutations=100,
                           verbose=0,
                           seed=1234):
    
    dat = get_data(f, l, cv=cv, testsize=testsize, seed=seed)
    model = tf.keras.models.load_model(model_dir)
    
    ref_error = model.evaluate(dat["data"],dat["labels"],verbose=0)
    mae = ref_error[1]
    mse = ref_error[2]
    
    errors_wo_feature = []
    for feat_id,feat_dat in enumerate(dat['feature_names']):
        if verbose:
            print('Processing feature  %s' % feat_dat)
        n_errors = []
        features = dat["data"].copy()
        for i in np.arange(n_permutations):
            features[:,feat_id] = np.random.permutation(features[:,feat_id])
            error = model.evaluate(features,dat["labels"],verbose=0)            
            n_errors.append(error)     
        errors_wo_feature.append(n_errors)
    sqrt_errors_wo_feature = np.sqrt(np.array(errors_wo_feature))

    delta_accs = sqrt_errors_wo_feature - np.sqrt(ref_error)    
    delta_accs_means = np.mean(delta_accs,axis=1)
    delta_accs_stds = np.std(delta_accs,axis=1)
    
    # standardize importance:
    importance = delta_accs_means[:,0]/np.sum(delta_accs_means[:,0])
    imp_features = dat['feature_names'][np.where(importance > 0.001)[0]]
    imp_features_indx = np.where(importance > 0.001)[0]
    
    for feat_id,feat_dat in enumerate(dat['feature_names']):
        print(feat_dat, importance[feat_id])

def run_predict(f,
                model,
                rescale_factor,
                rescale_labels,
                clim_variable):
    rescale_feat = f * rescale_factor
    dat = get_data(rescale_feat,testsize=0)
    predictions = model.predict(dat['data'])
    return rescale_MAT_data(predictions,rescale=rescale_labels,reverse=True,clim_variable=clim_variable)

