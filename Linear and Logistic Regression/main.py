import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import linear
import ridge
import classification

args = sys.argv[0:]

train_path = args[1].split("=", 1)[1]
val_path = args[2].split("=", 1)[1]
test_path = args[3].split("=", 1)[1]
out_path = args[4].split("=", 1)[1]
section = int(args[5].split("=", 1)[1])

# Reading Dataset

# Training set
df_train = pd.read_csv(train_path, header = None)

sample_name_train = df_train[0].to_numpy()
sample_name_train = np.reshape(sample_name_train,(sample_name_train.shape[0],1))
y_train = df_train[1].to_numpy()
y_train = np.reshape(y_train,(y_train.shape[0],1))
features_train = df_train.iloc[:,2:].to_numpy()

X_train = np.hstack((np.ones((np.shape(features_train)[0],1)), features_train))

# Validation set
df_validation = pd.read_csv(val_path, header = None)

sample_name_validation = df_validation[0].to_numpy()
y_validation = df_validation[1].to_numpy()
y_validation = np.reshape(y_validation,(y_validation.shape[0],1))
features_validation = df_validation.iloc[:,2:].to_numpy()

X_validation = np.hstack((np.ones((np.shape(features_validation)[0],1)), features_validation))

# Test set
df_test = pd.read_csv(test_path, header = None)

sample_name_test = df_test[0].to_numpy()
sample_name_test = np.reshape(sample_name_test,(sample_name_test.shape[0],1))
features_test = df_test.iloc[:,1:].to_numpy()
X_test = np.hstack((np.ones((np.shape(features_test)[0],1)), features_test))


if(section == 1):
    w, MSE_t, MSE_v, MSE_relative = linear.gradient_descent_withE(X_train, y_train, X_validation, y_validation, 0.001, 5000, 1e-6, 0)
    
    mse_train = np.sum((X_train@w-y_train)**2)/X_train.shape[0]
    mae_train = np.sum(np.abs(X_train@w-y_train))/X_train.shape[0]
    print("MSE for training set: ", mse_train)
    print("MAE for training set: ", mae_train)
    mse_validation = np.sum((X_validation@w-y_validation)**2)/X_validation.shape[0]
    mae_validation = np.sum(np.abs(X_validation@w-y_validation))/X_validation.shape[0]
    print("MSE for validation set: ", mse_validation)
    print("MAE for validation set: ", mae_validation)
    
    
    
    y_test = X_test@w
    test_out = np.hstack((sample_name_test, y_test))
   
    DF = pd.DataFrame(test_out)
    DF.to_csv(out_path+"/test_out.csv", header = False, index = False)
        
elif(section == 2):
    theta, rMSE_t, rMSE_v = ridge.ridge_gradient_descent(X_train, y_train, X_validation, y_validation, 100, 0.00001, 5000, 1e-5, 0)
    
    rmse_train = np.sum((X_train@theta-y_train)**2)/X_train.shape[0]
    rmae_train = np.sum(np.abs(X_train@theta-y_train))/X_train.shape[0]

    print("MSE for training set: ", rmse_train)
    print("MAE for training set: ", rmae_train)

    rmse_validation = np.sum((X_validation@theta-y_validation)**2)/X_validation.shape[0]
    rmae_validation = np.sum(np.abs(X_validation@theta-y_validation))/X_validation.shape[0]

    print("MSE for validation set: ", rmse_validation)
    print("MAE for validation set: ", rmae_validation)

    y_test = X_test@theta
    test_out = np.hstack((sample_name_test, y_test))
    
    DF = pd.DataFrame(test_out)
    DF.to_csv(out_path+"/test_out.csv", header = False, index = False)

elif(section == 5):
    theta, J_t, J_v = classification.classification_gradient_descent(X_train, y_train, X_validation, y_validation, 0.001, 5000, 1e-6)
    print("accuracy of training set", classification.accuracy(y_train.ravel(), classification.predict_classification(X_train, theta)))
    print("accuracy of validation set", classification.accuracy(y_validation.ravel(), classification.predict_classification(X_validation, theta)))
    
    y_test = classification.predict_classification(X_test, theta)
    test_out = np.hstack((sample_name_test, np.reshape(y_test,(len(y_test),1))))
    
    DF = pd.DataFrame(test_out)
    DF.to_csv(out_path+"/test_out.csv", header = False, index = False)
else:
    print("wrong section")
    
        
# main.py --train_path=train.csv --val_path=validation.csv --test_path=train.csv --out_path=test_out.csv --section=1





















# print(sys.argv)
