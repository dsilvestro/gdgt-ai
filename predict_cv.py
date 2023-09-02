import os, sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
gdgt_ai_path = "specify path to gdgt-ai"
sys.path.insert(0, gdgt_ai_path)
import gdgt

model_wd = "your path to gdgt-ai/trained_models"
wd = "your path to gdgt-ai/unlabeled_data"



def main(data_indx,  maf_nn=None, mat_nn=None):

    test_data_files = [
        os.path.join(wd, 'Raberg2022_all_data_unlabeled.txt'), # 0
        os.path.join(wd, "Salt_Lake_So_unlabeled.txt"), # 1
        os.path.join(wd, "Dearing_2021_Site625_unlabeled.txt"), # 2
        os.path.join(wd, "Amazon_GDGTs_unlabeled.txt"), # 3
        os.path.join(wd, "Garelick2022_unlabeled.txt"), # 4
        os.path.join(wd, "Hank_core2018_unlabeled.txt"), # 5
        os.path.join(wd, "Feakins_2019_unlabeled.txt"), # 6

        os.path.join(wd, "Weinan_loess_Tang_2017_unlabeled.txt"), # 7
        os.path.join(wd, "Lingtai_Loess_Fuchs_2022_unlabeled.txt"), # 8
        os.path.join(wd, "Xifeng_loess_section_Lu_2019_unlabeled.txt"), # 9
        os.path.join(wd, "Lantian_loess_Lu_2016_unlabeled.txt"), # 10

        os.path.join(wd, "Col17c_Colonia.txt"),  # 11
        os.path.join(wd, "CO14_unlabeled.txt"),  # 12

        os.path.join(wd, "soil_peat_features.txt_rnd.txt"), # 13 training data
        os.path.join(wd, "Zhao_2023_data.txt")  # 14 Zhao 2023

    ]

    brGDGDT_columnIDs = [
        'GDGT',
        'I',
        'I',
        'I',
        'I',
        'I',
        'b-',

        'I',
        'I',
        'I',
        'I',

        'I',
        'I',

        'I',
        'I'

    ]

    test_data_file = test_data_files[data_indx]
    brGDGDT_columnID = brGDGDT_columnIDs[data_indx]
    outname = os.path.basename(test_data_file).split(".txt")[0]
    print("\n\nRunning dataset:", test_data_file)
    nn_label_rescaler = 10

    # test_data_file = os.path.join(wd, "Dearing_2021_Site925_unlabeled.txt")
    # brGDGDT_columnID='I'
    # outname = "Dearing_2021_Site925_unlabeled"

    #---- RELOAD ORIGINAL MODEL ----#
    #---- train/test BNN with cross validation ----#
    # prep training data
    data = gdgt.GDGTdata(test_data_file,
                         brGDGDT_columnID=brGDGDT_columnID,
                         normalize_features=True,
                         sep='\t',
                         site_columnID='Sample'
                         )


    if mat_nn is None:
        retrain_nn = True
    else:
        retrain_nn = False
        maf_model = maf_nn
        mat_model = mat_nn

    ## start regular NN
    if retrain_nn:
        import matplotlib.pyplot as plt
        loss = 'mse' # 'log_cosh' #'mae'
        variable = "MAF"
        training_data_file = '/Users/dsilvestro/Software/gdgt-ai/R1/temperature_model_%s/terr_features.txt' % variable

        training_data = gdgt.GDGTdata(training_data_file,
                                      brGDGDT_columnID='fI', # keyword identifying brGDGT columns in the table
                                      CLIM_columnID=variable, # set to None is unlabeled data (for prediction)
                                      normalize_features=True,
                                      sep='\t',
                                      site_columnID="Sample")

        f = training_data._features.to_numpy()
        l = training_data._lCLIM.to_numpy()

        # randomize order
        indx = np.random.choice(np.arange(len(l)), len(l), replace=False)
        f = f[indx,:]
        l = l[indx]

        architecture = [tf.keras.layers.Flatten(input_shape=[f.shape[1]])]
        architecture.append(tf.keras.layers.Dense(128, activation='relu', use_bias=True))
        architecture.append(tf.keras.layers.Dense(64, activation='relu', use_bias=True))
        architecture.append(tf.keras.layers.Dense(32, activation='relu'))
        architecture.append(tf.keras.layers.Dense(1))
        maf_model = tf.keras.Sequential(architecture)
        optimizer = "adam"  # "adam" or tf.keras.optimizers.RMSprop(0.001)
        maf_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[loss])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_' + loss, patience=10,restore_best_weights=True)
        history = maf_model.fit(f, l / nn_label_rescaler,
                            epochs=500,
                            validation_split=0.2,
                            # batch_size=1000,
                            # validation_data=(dat['test_data'], dat['test_labels']),
                            verbose=1,
                            callbacks=[early_stop]
                            )


        fl = maf_model.predict(f).flatten() * nn_label_rescaler

        plt.scatter(l, fl, alpha=0.3)
        plt.plot(l, l, color="orange")
        title = "sig: %s" % np.round(np.mean(np.abs(fl-l)), 3)
        plt.gca().set_title(title, fontweight="bold", fontsize=10)
        plt.show()
        # plot residuals
        plt.scatter(l, fl-l)

        #
        #

        variable = "MAT"
        training_data_file = '/Users/dsilvestro/Software/gdgt-ai/R1/temperature_model_%s/terr_features.txt' % variable

        training_data = gdgt.GDGTdata(training_data_file,
                                      brGDGDT_columnID='fI', # keyword identifying brGDGT columns in the table
                                      CLIM_columnID=variable, # set to None is unlabeled data (for prediction)
                                      normalize_features=True,
                                      sep='\t',
                                      site_columnID="Sample")

        f = training_data._features.to_numpy()
        l = training_data._lCLIM.to_numpy()
        # randomize order
        indx = np.random.choice(np.arange(len(l)), len(l), replace=False)
        f = f[indx,:]
        l = l[indx]

        architecture = [tf.keras.layers.Flatten(input_shape=[f.shape[1]])]
        architecture.append(tf.keras.layers.Dense(128, activation='relu', use_bias=True))
        architecture.append(tf.keras.layers.Dense(64, activation='relu', use_bias=True))
        architecture.append(tf.keras.layers.Dense(32, activation='relu'))
        architecture.append(tf.keras.layers.Dense(1))
        mat_model = tf.keras.Sequential(architecture)
        optimizer = "adam"  # "adam" or tf.keras.optimizers.RMSprop(0.001)
        mat_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[loss])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_' + loss, patience=10,restore_best_weights=True)
        history = mat_model.fit(f, l / nn_label_rescaler,
                            epochs=500,
                            validation_split=0.1,
                            # batch_size=1000,
                            # validation_data=(dat['test_data'], dat['test_labels']),
                            verbose=1,
                            callbacks=[early_stop]
                            )

        fp = mat_model.predict(f)

    ## end regular NN


    # MAF terr peat model
    maf_list_ = []
    for cv in range(5):
        trained_model_filename = os.path.join(model_wd, 'MAF/bnn_logs/bnn_cv%s_empErr_p1_h0_l12_4_s1_binf_1234.pkl' % cv)
        model_settings_filename = os.path.join(model_wd, 'MAF/MAF_model_settings.pkl')
        # build a model based on pre-trained models
        model_settings = gdgt.load_model_settings(model_settings_filename)

        maf_ = gdgt.get_posterior_predictions(trained_model_filename,
                                             data,
                                             model_settings,
                                             outname="MAF",
                                             save_predictions=False)

        maf_list_.append(maf_['res_tbl'])

    maf_array = np.array(maf_list_)
    maf_array_avg_ = np.mean(maf_array[:,:,1:], 0)
    maf_list_res_ = maf_list_[0]
    maf_list_res_['prediction'] = maf_array_avg_[:,0]
    maf_list_res_['std.err'] = maf_array_avg_[:,1]

    res_tbl = pd.concat([maf_list_res_], axis=1)
    res_tbl.columns = ['Sample','MAFterr_peat.pred','MAFterr_peat.std.err']



    # MAT terr peat model
    maf_list_ = []
    for cv in range(5):
        trained_model_filename = os.path.join(model_wd, 'MAT/bnn_logs/bnn_cv%s_empErr_p1_h0_l12_4_s1_binf_1234.pkl' % cv)
        model_settings_filename = os.path.join(model_wd, 'MAT/MAT_model_settings.pkl')
        # build a model based on pre-trained models
        model_settings = gdgt.load_model_settings(model_settings_filename)

        maf_ = gdgt.get_posterior_predictions(trained_model_filename,
                                             data,
                                             model_settings,
                                             outname="MAT",
                                             save_predictions=False)

        maf_list_.append(maf_['res_tbl'])

    maf_array = np.array(maf_list_)
    maf_array_avg_ = np.mean(maf_array[:,:,1:], 0)
    maf_list_res_ = maf_list_[0]
    maf_list_res_['prediction'] = maf_array_avg_[:,0]
    maf_list_res_['std.err'] = maf_array_avg_[:,1]

    p = pd.concat(
        [ maf_list_res_[['prediction', 'std.err']]], axis=1)
    p.columns = ['MATterr_peat.pred', 'MATterr_peat.std.err']


    res_tbl = pd.concat([res_tbl,p], axis=1)


    ##### ADD NN models
    y_train = maf_model.predict(data._features.to_numpy(), verbose=1) * nn_label_rescaler
    p = pd.DataFrame(y_train)
    p.columns = ['NN-MAF']
    res_tbl = pd.concat([res_tbl, p], axis=1)

    y_train = mat_model.predict(data._features.to_numpy(), verbose=1) * nn_label_rescaler
    p = pd.DataFrame(y_train)
    p.columns = ['NN-MAT']
    res_tbl = pd.concat([res_tbl, p], axis=1)
    
    ### Save results
    res_tbl.to_csv(os.path.join(wd, 'Predictions_MAT_MAF_empErr_%s.txt' % outname), sep='\t')
    print("Results saved to:", wd, 'Predictions_MAT_MAF_empErr_%s.txt' % outname)
    return maf_model, mat_model


if __name__ == '__main__':
    maf_model, mat_model = None, None
    for i in range(15):
        print(i)
        maf_model, mat_model = main(i, maf_nn=maf_model, mat_nn=mat_model)

