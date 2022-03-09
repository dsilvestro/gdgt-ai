# if necessary, change working directory
wd ='your_path/gdgt-ai-main/'
import os
os.chdir(wd)

# load GDGT-AI library
import gdgt

training_data_file = 'data/MAT_training_data.csv'

#---- train/test BNN with cross validation ----#
# prep training data
training_data = gdgt.GDGTdata(training_data_file,
                              brGDGDT_columnID='brGDGT', # keyword identifying brGDGT columns in the table
                              CLIM_columnID='MAT', # set to None is unlabeled data (for prediction)
                              normalize_features=True,
                              )

# set up matnn model
model = gdgt.NNmodel(training_data)
# save model settings
model.save_model_settings(filename='MAT_model_settings.pkl')

p = model.cross_validation_bnn(n_layers=[12,4],
                               n_iteration=5000,
                               sampling_f=50,
                               n_post_samples=1000,
                               parallel_run=True, # run on multiple CPUs if set True
                               plot_predictions=True,
                               rescale_labels=False)


#---- re-train a model based on all data ----#
trained_model_filename = model.train_bnn(n_layers=[12,4],
                                         n_iteration=5000,
                                         sampling_f=50,
                                         n_post_samples=1000,
                                         testsize=0)

# predict all data based on production model (here using a model pre-trained through a longer run)
trained_model_filename = 'data/pretrained_bnn_MAT_model.pkl' # 

res = gdgt.get_posterior_predictions(trained_model_filename, 
                                     training_data,
                                     model)

# add true MAT values for plotting
res['true_MAT'] = training_data._lCLIM

# plot results
gdgt.plot_bnn(res, 
              variable_name='MAT',
              filename='data/Predicted_MAT_training_data.pdf',
              reference_label='true_MAT')
