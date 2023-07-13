import gdgt
import os
wd = "/Users/dsilvestro/Software/gdgt-ai/R1/data/"
training_data_file = os.path.join(wd, 'Raberg2022_all_data.csv')

#---- RELOAD ORIGINAL MODEL ----#
#---- train/test BNN with cross validation ----#
# prep training data
training_data = gdgt.GDGTdata(training_data_file,  
                              brGDGDT_columnID='brGDGT', # keyword identifying brGDGT columns in the table
                              CLIM_columnID='MAF',
                              normalize_features=True,
                              site_columnID='Sample_name'
                              )

#----- PREDICT BASED ON PRE-TRAINED MODEL ----#
# predict all data based on pre-trained model
trained_model_filename = os.path.join(wd, 'pretrained_bnn_MAT_model.pkl')
model = gdgt.NNmodel(training_data)
res = gdgt.get_posterior_predictions(trained_model_filename, training_data, model)
# add true MAT values for plotting
res['true_MAF'] = training_data._lCLIM
# plot results
gdgt.plot_bnn(res,
              variable_name='MAF',
              filename=os.path.join(wd, 'Predicted_MAF_training_data.pdf'),
              reference_label='true_MAF')

#----- RE-TRAIN MODEL ----#
# set up matnn model
model = gdgt.NNmodel(training_data)
# save model settings
model.save_model_settings(filename='MAF_model_settings.pkl')

p = model.cross_validation_bnn(n_layers=[12,4],
                               n_iteration=105000,
                               sampling_f=50,
                               n_post_samples=1000,
                               parallel_run=True, # run on multiple CPUs if set True
                               plot_predictions=True,
                               rescale_labels=False)


