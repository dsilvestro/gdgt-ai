# GDGT-AI - estimation of climatic variables using GDGT data

The GDGT-AI library requires the following Python (>= v.3.8) libraries:

pandas ~= '1.1.5'  
numpy ~= '1.19.5'  
scipy ~= '1.5.4'  
tensorflow ~= '2.4.1'  

### Tutorial (in progress)

**Runners** 

`01_train_bnn_model.py` trains Bayesian neural networks (BNNs) using cross validation. 

`02_predict_bnn.py` predicts unlabeled data using a trained BNN. 

Example files are provided in the `data` directory. 