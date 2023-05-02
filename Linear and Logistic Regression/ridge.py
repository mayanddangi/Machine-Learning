import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, r_regression,  mutual_info_regression

'''
    @input params
        Training Data
            X_t, y_t
        Validation Data
            X_v, y_v
        steps     : learning rates
        max_it    : max no. of iteration
        reltol    : threshold w.r.t relative change in cost function of validation data
        threshold : threshold w.r.t to square error in training data
        
    @output params
        w     : weights
        MSE_t : MSE of training data in each iteration
        MSE_v : MSE of validation data in each iteration
'''
def ridge_gradient_descent(X_t, y_t, X_v, y_v, lamda, steps, max_it, reltol, threshold):
    
    #Initalising Parameters
    N_t = X_t.shape[0]
    N_v = X_v.shape[0]
    d1 = X_t.shape[1]
#     theta = np.zeros([d1,1])
    theta = np.random.rand(d1,1)/10
    XT = np.transpose(X_t)
    XT_X = np.matmul(XT,X_t)
    
    MSE_t = []
    MSE_v = []
   
    # Iterative loop for minimizing cost function i.e. MSE     
    for i in range(0,max_it):
        
        # calculating gradient
        grad = 2*(np.matmul(XT_X, theta) - np.matmul(XT, y_t) + lamda*theta)
#         norm_grad = grad/np.sqrt(np.sum(grad**2))
        norm_grad = grad
        
        # updating weights    
        theta -= steps*norm_grad
        
        # estimating training error
        e_t = np.matmul(X_t,theta) - y_t
        MSEin_t = np.matmul(np.transpose(e_t),e_t)[0][0] + lamda*(theta.T@theta)[0][0]
#         MSEin_t = np.sum(e_t**2) + np.sum(theta**2)
        MSE_t.append(MSEin_t)
        
        # estimating validation error
        e_v = np.matmul(X_v,theta) - y_v
        MSEin_v = np.matmul(np.transpose(e_v),e_v)[0][0] + lamda*(theta.T@theta)[0][0]
#         MSEin_v = np.sum(e_v**2) + np.sum(theta**2)
        MSE_v.append(MSEin_v)
        
        # going out from loop if threshold has reached
        if(i > 1):
            if(np.abs(MSE_v[-2] - MSEin_v)/MSE_v[-2]<reltol):
                print(np.abs(MSE_v[-2] - MSEin_v)/MSE_v[-2])
                break
            
    
    return theta, MSE_t, MSE_v