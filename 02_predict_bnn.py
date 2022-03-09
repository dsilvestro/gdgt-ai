import numpy as np
import pandas as pd
import gdgt
print(gdgt.__version__)


#---- predict unlabeled data ----#
# here using a model pre-trained through a longer run:
unlab_infile = 'data/unlabeled_data.csv'
model_settings_filename = 'data/pretrained_MAT_model_settings.pkl' 
trained_model_filename = 'data/pretrained_bnn_MAT_model.pkl' 

unlabeled_data = gdgt.GDGTdata(unlab_infile, brGDGDT_columnID='brGDGT')

# build a model based on pre-trained models
model_settings = gdgt.load_model_settings(model_settings_filename)

res = gdgt.get_posterior_predictions(trained_model_filename, 
                                     unlabeled_data,
                                     model_settings
                               )


gdgt.plot_bnn(res, 
              variable_name='MAT',
              filename='data/Predicted_unlabeled_data.pdf')


