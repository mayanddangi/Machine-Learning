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

#################################################################################
#
#################################################################################

def get_feature(path):
    img = cv2.imread(path)
    return img.flatten()

print("__________Reading data__________")
person_path = glob.glob(os.path.join(train_path+"/person/", "*.png"))
not_person_path = glob.glob(os.path.join(train_path+"/airplane/", "*.png")) + glob.glob(os.path.join(train_path+"/car/", "*.png")) + glob.glob(os.path.join(train_path+"/dog/", "*.png"))

# Reading Training Dataset
X_per = None
i=0
for img in person_path:
    if(i==0):
        X_per = get_feature(person_path[0])
        i+=1
    else:
        X_per = np.vstack((X_per, get_feature(img)))
        
X_not_per = None
i=0
for img in not_person_path:
    if(i==0):
        X_not_per = get_feature(not_person_path[0])
        i+=1
    else:
        X_not_per = np.vstack((X_not_per, get_feature(img)))
        
print(f'Dimension of X_per: {X_per.shape}\nDimension of X_not_per: {X_not_per.shape}')

X_train = np.vstack((X_per, X_not_per))
y_train = np.vstack((np.ones((len(X_per), 1)), np.zeros((len(X_not_per), 1))))      # y_per + y_not_per
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
        
y_test = np.asarray([0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,0,0,1])
image_id = np.reshape(image_id, (len(image_id),1))
# print(image_id)

print(f'Dimension of X_test: {X_test.shape}\n')

########################################################################################
#
########################################################################################

#--------------------------------------------------------------------------------------#
## (a) -- Decision Tree from scratch
print("__________3.1 (a) start:___________")
T = dt(10, 7, 'gini')
T.fit(X_train, y_train)
y = T.predict(X_test)
y = np.reshape(y,(len(y),1))

test_out = np.hstack((image_id, y))
DF = pd.DataFrame(test_out)
DF.to_csv(out_path+"/test_31a.csv", header = False, index = False)
print("__________3.1 (a) end\n\n")
#--------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------#
## (b) -- Decision Tree sklearn
print("__________3.1 (b) start:___________")
start = time.time()
clf = DecisionTreeClassifier(max_depth = 10, criterion = 'gini', min_samples_split = 7)
clf = clf.fit(X_train,y_train)
y = clf.predict(X_test)
y = np.int_(np.reshape(y,(len(y),1)))

test_out = np.hstack((image_id, y))
DF = pd.DataFrame(test_out)
DF.to_csv(out_path+"/test_31b.csv", header = False, index = False)
print(f'execution time: {time.time()-start} sec\n__________3.1 (b) end\n\n')
#--------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------#
## (c) -- Decision Tree best 10 feature
print("__________3.1 (c) start:___________")
start = time.time()
selector = SelectKBest(mutual_info_regression, k=10)
X_train_new = selector.fit_transform(X_train, y_train.ravel())
X_test_new = X_test[:,selector.get_support()]


clf = DecisionTreeClassifier(max_depth = 5, criterion = 'gini', min_samples_split = 9)
clf = clf.fit(X_train_new,y_train)
y = clf.predict(X_test_new)
y = np.int_(np.reshape(y,(len(y),1)))

test_out = np.hstack((image_id, y))
DF = pd.DataFrame(test_out)
DF.to_csv(out_path+"/test_31c.csv", header = False, index = False)
print(f'execution time: {time.time()-start} sec\n__________3.1 (c) end\n\n')
#--------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------#
## (d) -- Decision Tree Post Pruning
print("__________3.1 (d) start:___________")
start = time.time()
clf = DecisionTreeClassifier(ccp_alpha=0.0027354, criterion = 'entropy')
clf.fit(X_train, y_train)
y = clf.predict(X_test)
y = np.int_(np.reshape(y,(len(y),1)))

test_out = np.hstack((image_id, y))
DF = pd.DataFrame(test_out)
DF.to_csv(out_path+"/test_31d.csv", header = False, index = False)
print(f'execution time: {time.time()-start} sec\n__________3.1 (d) end\n\n')
#--------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------#
## (e) -- Random Forest
print("__________3.1 (e) start:___________")
start = time.time()
# clf = RandomForestClassifier(criterion='entropy', min_samples_split= 5, n_estimators=150, max_depth = 10)
clf = RandomForestClassifier(criterion='entropy', min_samples_split= 5, n_estimators=200)
clf.fit(X_train, y_train.ravel())
y = clf.predict(X_test)
y = np.int_(np.reshape(y,(len(y),1)))

test_out = np.hstack((image_id, y))
DF = pd.DataFrame(test_out)
DF.to_csv(out_path+"/test_31e.csv", header = False, index = False)
print(f'execution time: {time.time()-start} sec\n__________3.1 (e) end\n\n')
#--------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------#
## (f) -- XGBoost
print("__________3.1 (f) start:___________")
start = time.time()
clf = XGBClassifier(tree_method='hist', max_depth = 9, n_estimators = 50, subsample = 0.4)
clf.fit(X_train, y_train.ravel())
y = clf.predict(X_test)
y = np.int_(np.reshape(y,(len(y),1)))

test_out = np.hstack((image_id, y))
DF = pd.DataFrame(test_out)
DF.to_csv(out_path+"/test_31f.csv", header = False, index = False)
print(f'execution time: {time.time()-start} sec\n__________3.1 (f) end\n\n')
#--------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------#
## (h) competitive part
print("__________3.1 (h) start:___________")
start = time.time()
DF.to_csv(out_path+"/test_31h.csv", header = False, index = False)
print(f'execution time: {time.time()-start} sec\n__________3.1 (h) end\n\n')
#--------------------------------------------------------------------------------------#



# python main_binary.py --train_path=./data/train --test_path=./data/test_sample --out_path=./out
