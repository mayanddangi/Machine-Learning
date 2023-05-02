from typing import List
import numpy as np
import qpsolvers
from kernel import *

import numpy as np
import pandas as pd
import qpsolvers as qp
import tqdm
import math
class Trainer:
    def __init__(self,kernel,C=None,**kwargs) -> None:
        self.kernel = kernel
        self.kwargs = kwargs
        self.C=C
        if(self.C is not None):
            self.C = float(self.C)
        self.support_vectors:List[np.ndarray] = []
        
        self.gamma = 1.0
        self.deg = 1
        for key, value in kwargs.items():
            if(key=='gamma'):
                self.gamma = value
            if(key=='degree'):
                self.deg = value
            if(key=='offset'):
                self.offset = value
        
    
    def fit(self, train_data_path:str)-> None:
        #store the support vectors in self.support_vectors
        
        # Read training dataset
        df_train = pd.read_csv(train_data_path)

        sample_name_train = df_train['Unnamed: 0'].to_numpy()
        sample_name_train = np.reshape(sample_name_train,(sample_name_train.shape[0],1))

        y_train = df_train['y'].to_numpy()
        y_train = np.reshape(y_train,(y_train.shape[0],1))

        self.X_train = df_train.iloc[:,1:2049].to_numpy()

        
        # determining required parameters
        if(self.kernel  == 'linear'):
            kernel_matrix = linear(self.X_train)
        elif(self.kernel == 'polynomial'):
            kernel_matrix = polynomial(self.X_train, degree = self.deg, gamma = self.gamma)
        elif(self.kernel == 'rbf'):
            kernel_matrix = rbf(self.X_train, Z = self.X_train, gamma = self.gamma)
        elif(self.kernel == 'sigmoid'):
            kernel_matrix = sigmoid(self.X_train, Z = self.X_train, gamma = self.gamma, offset = self.offset)
        elif(self.kernel == 'laplacian'):
            kernel_matrix = laplacian(self.X_train, Z = self.X_train, gamma = self.gamma)
        else:
            raise ValueError('invalid kernel')
            
        # QP problem
        N = len(y_train)
        
        self.y = np.ones(np.shape(y_train))
        self.y[y_train == 0] = -1

        P = np.matmul(self.y, self.y.T)*kernel_matrix
        q = -np.ones(N)
        G = np.vstack((-np.eye(N), np.eye(N)))
        h = np.vstack((np.zeros((N, 1)), np.ones((N, 1))*self.C))
        A = self.y.reshape(1, N)*1.0
        b = np.array([0.0])

        self.alpha = qp.solve_qp(P, q, G, h, A, b, solver='cvxopt')        
        self.alpha = np.reshape(self.alpha,(self.alpha.shape[0],1))
        
        # Support vectors
        
        self.b = np.mean(self.y - np.matmul(kernel_matrix, self.y*self.alpha))

    
    def predict(self, test_data_path:str)-> np.ndarray:
        #TODO: implement
        
        # Read data
        df_test = pd.read_csv(test_data_path)

        sample_name_test = df_test['Unnamed: 0'].to_numpy()
        sample_name_test = np.reshape(sample_name_test,(sample_name_test.shape[0],1))
        
        # y_test = df_test['y'].to_numpy()
        # y_test = np.reshape(y_test,(y_test.shape[0],1))

        X_test = df_test.iloc[:,1:2049].to_numpy()

        
        # Determine kernel_matrix
        if(self.kernel  == 'linear'):
            kernel_matrix = linear(X_test)
        elif(self.kernel == 'polynomial'):
            kernel_matrix = polynomial(X_test, Z = self.X_train, degree = self.deg, gamma = self.gamma)
        elif(self.kernel == 'rbf'):
            kernel_matrix = rbf(X_test, Z = self.X_train, gamma = self.gamma)
        elif(self.kernel == 'sigmoid'):
            kernel_matrix = sigmoid(X_test, Z = self.X_train, gamma = self.gamma, offset = self.offset)
        elif(self.kernel == 'laplacian'):
            kernel_matrix = laplacian(X_test, Z = self.X_train, gamma = self.gamma)
        else:
            raise ValueError('Invalid Kernel')
            
        y_pred = np.dot(kernel_matrix, self.y*self.alpha) + self.b
        y_pred[y_pred >= 0] = 1
        y_pred[y_pred < 0] = 0
        
        return y_pred
    
    def fit_multi(self, X_train, y_train)-> None:
        #store the support vectors in self.support_vectors
        
        # Read training dataset
        self.X_train = X_train
        self.y = y_train
        N = len(y_train)

#         print(f'Shape of X_train: {self.X_train.shape} \n Shape of y_train: {y_train.shape}')
        
        # determining required parameters
        if(self.kernel  == 'linear'):
            kernel_matrix = linear(self.X_train)
        elif(self.kernel == 'polynomial'):
            kernel_matrix = polynomial(self.X_train, degree = self.deg, gamma = self.gamma)
        elif(self.kernel == 'rbf'):
            kernel_matrix = rbf(self.X_train, Z = self.X_train, gamma = self.gamma)
        elif(self.kernel == 'sigmoid'):
            kernel_matrix = sigmoid(self.X_train, Z = self.X_train, gamma = self.gamma, offset = self.offset)
        elif(self.kernel == 'laplacian'):
            kernel_matrix = laplacian(self.X_train, Z = self.X_train, gamma = self.gamma)
        else:
            raise ValueError('invalid kernel')
            

        P = np.matmul(self.y, self.y.T)*kernel_matrix
        q = -np.ones(N)
        G = np.vstack((-np.eye(N), np.eye(N)))
        h = np.vstack((np.zeros((N, 1)), np.ones((N, 1))*self.C))
        A = self.y.reshape(1, N)*1.0
        b = np.array([0.0])

        self.alpha = qp.solve_qp(P, q, G, h, A, b, solver='cvxopt')        
        self.alpha = np.reshape(self.alpha,(self.alpha.shape[0],1))
        
        # Support vectors
        
        self.b = np.mean(self.y - np.matmul(kernel_matrix, self.y*self.alpha))
        
        
    def predict_multi(self, X_test)-> np.ndarray:
        #Return the predicted labels as a numpy array of dimension n_samples on test data

        
        # Determine kernel_matrix
        if(self.kernel  == 'linear'):
            kernel_matrix = linear(X_test)
        elif(self.kernel == 'polynomial'):
            kernel_matrix = polynomial(X_test, Z = self.X_train, degree = self.deg, gamma = self.gamma)
        elif(self.kernel == 'rbf'):
            kernel_matrix = rbf(X_test, Z = self.X_train, gamma = self.gamma)
        elif(self.kernel == 'sigmoid'):
            kernel_matrix = sigmoid(X_test, Z = self.X_train, gamma = self.gamma, offset = self.offset)
        elif(self.kernel == 'laplacian'):
            kernel_matrix = laplacian(X_test, Z = self.X_train, gamma = self.gamma)
        else:
            raise ValueError('Invalid Kernel')

        y_pred = np.dot(kernel_matrix, self.y*self.alpha) + self.b
#         y_pred[y_pred >= 0] = 1
#         y_pred[y_pred < 0] = 0

        return y_pred