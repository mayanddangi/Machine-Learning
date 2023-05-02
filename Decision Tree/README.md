## Overview

**`decision_tree.py`** contains the class which is used to build, fit, predict and plot decision tree. The **`decisionTree`** class has the following methods:

 - `score(X, y):` return accuracy of model
 - `get_params`: return max_depth, min_samples_split and criterion
 - `set_params`: to set the parameters
 - `calculate_entropy(y)`: return entropy
 - `calculate_infogain(X,y,feature, threshold)`: return information gain
 - `_gini_index(y)`: return gini impurity
 - `_find_best_split_ginni(X,y)`: return the best feature and threshold to split the data based on the Gini impurity.
 - `bestFeature_ginni(X,y)`: return best feature and threshold(which is median of the label)
 - `bestFeature_a(X,y)`: return best feature based on information gain
 - `fit(self, train_data_path:str)`: Fit the decision tree to the binary training data X and y
 - `predict(self, test_data_path:str)`: Predict the binary class labels for the input data X
 - `growTree(X,y,depth)`: recursively grow tree
 - `plot()`: plot the decision tree using graphviz



## Dependencies
This code requires the following dependencies:

- numpy (for numerical operations)
- matplotlib (for visualization)
- graphviz (for visulaisation)
- sklearn
- pandas
- time
- os
- sys
- cv2 (reading image)
- glob
- tqdm

## Usage
For binary classification use main_binary.py with following script 

```
  python main_binary.py --train_path=./data/train --test_path=./data/test_sample --out_path=.
```

For multiclass classification use main_binary.py with following script 
```
python main_multi.py --train_path=./data/train --test_path=./data/test_sample --out_path=.
```

There are also two jupyter notebook seperate for multi and binary
