# GDGT-AI - estimation of climatic variables using GDGT data

This library implements neural network models to infer paleotemperatures from branched glycerol dialkyl glycerol tetraethers (brGDGTs). The method is described in [this paper](https://doi.org/10.1016/j.gca.2023.09.014).

The GDGT-AI library requires the following Python (>= v.3.10) libraries:

pandas ~= '1.1.5'  
numpy ~= '1.19.5'  
scipy ~= '1.5.4'  
tensorflow ~= '2.4.1'  

### Tutorial 

**Model training** 

The scripts `bnn_mat_model.py` and `bnn_maf_model.py` show how to import the **gdgt-ai** library and use it to train Bayesian neural networks (BNNs) using cross validation. The scripts 
train models to inferred mean annual temperature (MAT) and mean annual temperature above freezing (MAF). 
rely on the data available in the `training_data` directory. This step can take several hours and pre-trained BNN models are available in the `trained_models` directory. 


**Predictions**

The script `predict_unlabeled_data.py` shows how to predict MAT and MAF based on trained BNN models and using regular neural networks. the script loops over all files provided in the `unlabeled_data` directory and saves the BNN and NN predictions in tab-separated text files. The paths to the **gdgt-ai** library and input files can be specified in the first 15 lines of the script. 