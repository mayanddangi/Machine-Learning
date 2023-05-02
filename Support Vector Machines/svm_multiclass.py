from typing import List
import numpy as np
from svm_binary import Trainer
from kernel import *

import numpy as np
import pandas as pd
import qpsolvers as qp
import tqdm
import math
    
class Trainer_OVA:
    def __init__(self, kernel, C=None, n_classes = -1, **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        self.classes = None
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]
        
        self.gamma = 0.1
        
        for key, value in kwargs.items():
            if(key=='gamma'):
                self.gamma = value
        
    
    def _init_trainers(self):
        #TODO: implement
        #Initiate the svm trainers 
        pass
    
    def fit(self, train_data_path:str, max_iter=None)->None:
        #TODO: implement
        #Store the trained svms in self.svms
        
        # Read data
        df_train = pd.read_csv(train_data_path)
        
        y_train = df_train['y'].to_numpy()
        y_train = np.reshape(y_train,(y_train.shape[0],1))
        X_train = df_train.iloc[:,2:].to_numpy()

        self.classes = np.unique(y_train)
        self.n_classes = len(self.classes)
        
        for cls in range(0, self.n_classes):
            y_cls = np.where(y_train == self.classes[cls], 1, -1)
            T = Trainer(kernel = self.kernel, C = self.C, gamma = self.gamma)
            T.fit_multi(X_train, y_cls)
            self.svms.append(T)
    
   
    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels
        
        # Read dataset
        df_test = pd.read_csv(test_data_path)
        X_test = df_test.iloc[:,1:].to_numpy()
        
        # Predection
        y_pred = np.zeros((df_test.shape[0], self.n_classes))
        
        
        for cls in range(0, self.n_classes):
            y_pred[:,cls] = self.svms[cls].predict_multi(X_test).ravel()
        
        return self.classes[np.argmax(y_pred, axis = 1)]
        
        
        
class Trainer_OVO:
    def __init__(self, kernel, C=None, n_classes = -1, **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]
        
        self.gamma = 0.1
        
        for key, value in kwargs.items():
            if(key=='gamma'):
                self.gamma = value
    
    def _init_trainers(self):
        #TODO: implement
        #Initiate the svm trainers 
        pass
    
    def fit(self, train_data_path:str, max_iter=None)->None:
        #TODO: implement
        #Store the trained svms in self.svms

        # Read data
        df_train = pd.read_csv(train_data_path)
        
        y_train = df_train['y'].to_numpy()
        y_train = np.reshape(y_train,(y_train.shape[0],1))
        X_train = df_train.iloc[:,2:].to_numpy()

        # Obtain unique classes
        self.classes = np.unique(y_train)
        self.n_classes = len(self.classes)
        
        for i in range(0, self.n_classes):
            for j in range(i+1, self.n_classes):
                # Obtain X and y for class i and j
                indices = ((y_train == self.classes[i]) | (y_train == self.classes[j])).ravel()
                X_ij = X_train[indices]
                y_ij = y_train[indices]

                y_ij[y_ij == self.classes[i]] = -1
                y_ij[y_ij == self.classes[j]] = 1

                # Trainer object
                T = Trainer(kernel = 'rbf', C = self.C, gamma = self.gamma)
                T.fit_multi(X_ij, y_ij)
                self.svms.append([i, j, T])
        
    
    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels
        
        # Read data
        df_test = pd.read_csv(test_data_path)
        X_test = df_test.iloc[:,1:].to_numpy()

        # initialize y_pred
        y_pred = np.zeros((df_test.shape[0], self.n_classes))

        count = 0

       
        for pointer in range(0, len(self.svms)):
            i = self.svms[pointer][0]
            j = self.svms[pointer][1]

            y_pred_ij = self.svms[pointer][2].predict_multi(X_test)
            y_pred_ij = np.where(y_pred_ij < 0, self.classes[i], self.classes[j])

            y_pred[:, i] += np.where((y_pred_ij==self.classes[i]).ravel(), 1, 0)
            y_pred[:, j] += np.where((y_pred_ij==self.classes[j]).ravel(), 1, 0)

        return self.classes[np.argmax(y_pred, axis = 1)]
