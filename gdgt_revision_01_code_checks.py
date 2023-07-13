import gdgt
import os
wd = "/Users/dsilvestro/Software/gdgt-ai/R1/data/"
training_data_file = os.path.join(wd, 'BNN_South_America_GDGT.csv')

#---- RELOAD ORIGINAL MODEL ----#
#---- train/test BNN with cross validation ----#
# prep training data
training_data = gdgt.GDGTdata(training_data_file,
                              brGDGDT_columnID='brGDGT', # keyword identifying brGDGT columns in the table
                              CLIM_columnID='MAT', # set to None is unlabeled data (for prediction)
                              normalize_features=True,
                              )


# predict all data based on production model (here using a model pre-trained through a longer run)
trained_model_filename = os.path.join(wd, 'pretrained_bnn_MAT_model.pkl')
model = gdgt.NNmodel(training_data)
res = gdgt.get_posterior_predictions(trained_model_filename,
                                     training_data,
                                     model)

# add true MAT values for plotting
res['true_MAT'] = training_data._lCLIM

# plot results
gdgt.plot_bnn(res,
              variable_name='MAT',
              filename=os.path.join(wd, 'Predicted_MAT_training_data.pdf'),
              reference_label='true_MAT')

#----- RE-PREDICT UNALBELED TEST SET ----#
# here using a model pre-trained through a longer run:
unlab_infile = os.path.join(wd, 'Feakins_test_BNN.csv')
model_settings_filename = os.path.join(wd, 'pretrained_MAT_model_settings.pkl')
trained_model_filename = os.path.join(wd, 'pretrained_bnn_MAT_model.pkl')

unlabeled_data = gdgt.GDGTdata(unlab_infile, brGDGDT_columnID='brGDGT')

# build a model based on pre-trained models
model_settings = gdgt.load_model_settings(model_settings_filename)

res = gdgt.get_posterior_predictions(trained_model_filename,
                                     unlabeled_data,
                                     model_settings)

gdgt.plot_bnn(res,
              variable_name='MAF',
              filename=os.path.join(wd, 'Predicted_unlabeled_data.pdf'))


