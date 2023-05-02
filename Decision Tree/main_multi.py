############################################################################
#
############################################################################
# Import Libraries
import os
import time
import sys
import csv
import glob


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from decision_tree import decisionTree as dt

import graphviz
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, r_regression,  mutual_info_regression, SelectFromModel
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay, f1_score

#--------------------------------------------------------------------------------

args = sys.argv[0:]
train_path = args[1].split("=", 1)[1]
test_path = args[2].split("=", 1)[1]
out_path = args[3].split("=", 1)[1]

#--------------------------------------------------------------------------------
def get_feature(path):
    img = cv2.imread(path)
    return img.flatten()
#################################################################################
#
#################################################################################
person_path = glob.glob(os.path.join("./data/train/person/", "*.png"))
air_path = glob.glob(os.path.join("./data/train/airplane/", "*.png"))
dog_path = glob.glob(os.path.join("./data/train/dog/", "*.png"))
car_path = glob.glob(os.path.join("./data/train/car/", "*.png"))

X_per = None
i=0
for img in person_path:
    if(i==0):
        X_per = get_feature(person_path[0])
        i+=1
    else:
        X_per = np.vstack((X_per, get_feature(img)))
        
X_air = None
i=0
for img in air_path:
    if(i==0):
        X_air = get_feature(air_path[0])
        i+=1
    else:
        X_air = np.vstack((X_air, get_feature(img)))

X_car = None
i=0
for img in car_path:
    if(i==0):
        X_car = get_feature(car_path[0])
        i+=1
    else:
        X_car = np.vstack((X_car, get_feature(img)))

X_dog = None
i=0
for img in dog_path:
    if(i==0):
        X_dog = get_feature(dog_path[0])
        i+=1
    else:
        X_dog = np.vstack((X_dog, get_feature(img)))

print(f'Dimension of X_per: {X_per.shape}\nDimension of X_air: {X_air.shape}\nDimension of X_dog: {X_dog.shape}\nDimension of X_air: {X_car.shape}')

X_train = np.vstack((X_car, X_per, X_air, X_dog))
y_train = np.vstack((np.zeros((len(X_car), 1)), np.ones((len(X_per), 1)), 2*np.ones((len(X_air), 1)), 3*np.ones((len(X_dog), 1))))
print(f'Dimension of X_train: {X_train.shape}\nDimension of y_train: {y_train.shape}')

# Reading Test Dataset
path_test = sorted(glob.glob(os.path.join(test_path, "*.png")),key=len)

X_test = None
i=0
image_id = []
for img in path_test:
    if(i==0):
        X_test = get_feature(path_test[0])
        image_id.append(os.path.basename(img).split(".")[0])
        i+=1
    else:
        X_test = np.vstack((X_test, get_feature(img)))
        image_id.append(os.path.basename(img).split(".")[0])
        
image_id = np.reshape(image_id, (len(image_id),1))
# print(image_id)

print(f'Dimension of X_test: {X_test.shape}\n')

########################################################################################
#
########################################################################################

#--------------------------------------------------------------------------------------#
print("__________3.2 (a) start:___________")
clf = DecisionTreeClassifier(max_depth = 15, criterion = 'entropy', min_samples_split = 4)
clf = clf.fit(X_train,y_train)
y = clf.predict(X_test)
y = np.int_(np.reshape(y,(len(y),1)))

test_out = np.hstack((image_id, y))
DF = pd.DataFrame(test_out)
DF.to_csv(out_path+"/test_32a.csv", header = False, index = False)
print("__________3.2 (a) end\n\n")
#--------------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------------#
## (b) -- Decision Tree best 10 feature

print("__________3.2 (b) start:___________")
start = time.time()
selector = SelectKBest(mutual_info_regression, k=10)
X_train_new = selector.fit_transform(X_train, y_train.ravel())
X_test_new = X_test[:,selector.get_support()]

clf = DecisionTreeClassifier(max_depth = 15, criterion = 'entropy', min_samples_split = 4)
clf = clf.fit(X_train_new,y_train)
y = clf.predict(X_test_new)
y = np.int_(np.reshape(y,(len(y),1)))

test_out = np.hstack((image_id, y))
DF = pd.DataFrame(test_out)
DF.to_csv(out_path+"/test_32b.csv", header = False, index = False)
print(f'execution time: {time.time()-start} sec\n__________3.2 (b) end\n\n')
#--------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------#
## (c) -- Decision Tree Post Pruning
print("__________3.2 (c) start:___________")
start = time.time()
clf = DecisionTreeClassifier(ccp_alpha=0.00162255, criterion = 'entropy')
clf.fit(X_train, y_train)
y = clf.predict(X_test)
y = np.int_(np.reshape(y,(len(y),1)))

test_out = np.hstack((image_id, y))
DF = pd.DataFrame(test_out)
DF.to_csv(out_path+"/test_32c.csv", header = False, index = False)
print(f'execution time: {time.time()-start} sec\n__________3.2 (c) end\n\n')
#--------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------#
## (d) -- Random Forest
print("__________3.2 (d) start:___________")
start = time.time()
clf = RandomForestClassifier(criterion='entropy', min_samples_split= 5, n_estimators=200)
clf.fit(X_train, y_train.ravel())
y = clf.predict(X_test)
y = np.int_(np.reshape(y,(len(y),1)))

test_out = np.hstack((image_id, y))
DF = pd.DataFrame(test_out)
DF.to_csv(out_path+"/test_32d.csv", header = False, index = False)
print(f'execution time: {time.time()-start} sec\n__________3.2 (d) end\n\n')
#--------------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------------#
## (e) -- XGBoost
print("__________3.2 (e) start:___________")
start = time.time()
model = XGBClassifier(max_depth = 8, n_estimators = 50, subsample = 0.4)
model.fit(X_train, y_train)
y = model.predict(X_test)
y = np.int_(np.reshape(y,(len(y),1)))

test_out = np.hstack((image_id, y))
DF = pd.DataFrame(test_out)
DF.to_csv(out_path+"/test_32e.csv", header = False, index = False)
print(f'execution time: {time.time()-start} sec\n__________3.2 (e) end\n\n')
#--------------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------------#
## (h) competitive part
print("__________3.2 (h) start:___________")
start = time.time()
DF.to_csv(out_path+"/test_32h.csv", header = False, index = False)
print(f'execution time: {time.time()-start} sec\n__________3.2 (h) end\n\n')
#--------------------------------------------------------------------------------------#

# main_multi.py --train_path=./data/train --test_path=./data/test_sample --out_path=./out_multi
