import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, r_regression,  mutual_info_regression

def get_prob(X, theta):
    B = np.exp(np.matmul(X, theta) - np.max(np.matmul(X, theta)))
    norm = np.reshape((np.exp(-np.max(np.matmul(X, theta)))+np.sum(B, axis = 1)), (B.shape[0],1))
    B = B/norm
    return B

def getH(y_train, k):
    H = np.repeat(y_train, k, axis = 1)
    for i in range(1, k+1):
        H[H[:,i-1] !=i ,i-1] = 0
        H[:,i-1]=H[:,i-1]/i
    return H

def predict_classification(X, theta):
    prob = get_prob(X, theta)
    prob = np.hstack((prob, np.reshape((1-np.sum(prob, axis = 1)),(prob.shape[0],1))))
    return np.argmax(prob, axis=1)+1

def accuracy(y, y_hat):
    return np.sum(y==y_hat)/len(y)
'''
    @input params
        Training Data
            X_t, y_t
        Validation Data
            X_v, y_v
        steps     : learning rates
        max_it    : max no. of iteration
        reltol    : threshold w.r.t relative change in cost function of validation data
        
    @output params
        w     : weights
        J_t : Cost function of training data in each iteration
        J_v : Cost of validation data in each iteration
'''
def classification_gradient_descent(X_t, y_t, X_v, y_v, steps, max_it, reltol):
    
    #Initalising Parameters
    N = X_t.shape[0]
    d1 = X_t.shape[1]
    
    k = len(np.unique(y_t))-1
    
#     theta = np.random.random((d1,k))
    theta = np.zeros((d1,k))
    print(theta.shape)
    
        
    J_t = []
    J_v = []
    
    H = getH(y_t, k)
    H_v = getH(y_v, k)
    
    for i in range(0, max_it):  
        
        h_theta = get_prob(X_t,theta)
        # calcuate gradient
        grad = X_t.T@(h_theta - H)
        # weight update
        theta -= steps*grad
        
        # Cost Functions
        ## Train set
        J_ti = -np.sum(H*(np.log(h_theta)))
        J_t.append(J_ti)
        
        ## Validation set
        J_vi = -np.sum(H_v*(np.log(get_prob(X_v,theta))))
        J_v.append(J_vi)
        
        # going out from loop if validation cost function change is slow
        if(i > 1):
            if(np.abs(J_v[-2] - J_vi)/J_v[-2]<reltol):
                print(np.abs(J_v[-2] - J_vi)/J_v[-2])
                break
                
    return theta, J_t, J_v