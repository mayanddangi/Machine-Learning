import numpy as np
import time
import graphviz

class Node:
    def __init__(self, feature, threshold):
        self.parent = None
        self.left = None
        self.right = None
        self.feature = feature
        self.threshold = threshold
        self.isLeaf = False
        self.leaf_value = None
        
        
class decisionTree:
    def __init__(self, max_depth = 10, min_samples_split = 7, criterion = 'entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.criterion = criterion
                
    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
    
    def get_params(self, deep=True):
        return {'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'criterion': self.criterion}
    
    def set_params(self, **params):
        if 'max_depth' in params:
            self.max_depth = params['max_depth']
        if 'min_samples_split' in params:
            self.min_samples_split = params['min_samples_split']
        if 'criterion' in params:
            self.criterion = params['criterion']
        return self
            
    def calculate_entropy(self, y):
        H = 0
        if(len(y)!=0):
            p0 = len(y[y==0])/len(y)
            p1 = 1-p0
            if(p0!=0 and p0!=1):
                H = -(p0*np.log2(p0) + p1*np.log2(p1))
        return H

    def calculate_infogain(self, X, y, feature, threshold):
        # split dataset
        y_left = y[np.where(X[:, feature] <= threshold)]
        y_right = y[np.where(X[:, feature] > threshold)]

        H_left = self.calculate_entropy(y_left)
        H_right = self.calculate_entropy(y_right)
        H_node = self.calculate_entropy(y)

        p_left = len(y_left)/len(y)
        p_right = len(y_right)/len(y)

        IG = H_node - (p_left*H_left + p_right*H_right)

        return IG
    
    def _gini_index(self, y):
        """
        Calculate the Gini impurity of a set of labels.
        y: ndarray of shape (n_samples,), the target values.
        """
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        gini = 1 - np.sum(proportions ** 2)
        return gini
    

    def _find_best_split_ginni(self, X, y):
        """
        Find the best feature and threshold to split the data based on the Gini impurity.
        X: ndarray of shape (n_samples, n_features), the input samples.
        y: ndarray of shape (n_samples,), the target values.
        """
        N, d = X.shape
        best_gini = float('inf')
        best_feature = -1
        best_threshold = -1
        
        for i in range(d):
            for t in range(int(np.median(X[:, i]))):
                
                left = X[:, i] <= t
                right = X[:, i] > t
                
                left_gini = self._gini_index(y[left])
                right_gini = self._gini_index(y[right])

                gini = (np.sum(left) / N)*left_gini + (np.sum(right) / N)*right_gini

                if(gini < best_gini):
                    best_gini = gini
                    best_feature = i
                    best_threshold = t

        return best_feature, best_threshold
    
    def bestFeature_ginni(self, X, y):
        N, d  = X.shape
        
        best_gini = float('inf')
        best_feature = -1
        
        for i in range(d):
            left = X[:, i] <= np.median(X[:, i])
            right = X[:, i] > np.median(X[:, i])

            left_gini = self._gini_index(y[left])
            right_gini = self._gini_index(y[right])

            gini = (np.sum(left) / N)*left_gini + (np.sum(right) / N)*right_gini

            if(gini < best_gini):
                best_gini = gini
                best_feature = i

        return best_feature, np.median(X[:, best_feature])
    
    def bestFeature_a(self, X, y):
        N, d  = X.shape
        best_feature_no = -1

        IG_max = 0
        for i in range(d):
            IG = self.calculate_infogain(X, y, i, np.median(X[:, i]))
            if(IG > IG_max):
                IG_max = IG
                best_feature_no = i

        return best_feature_no

    def _find_best_split(self,X, y):
        N, d  = X.shape
        best_feature_no = -1
        best_threshold = -1
        IG_max = 0
        for i in range(d):
            for t in range(int(np.median(X[:, i]))):
                IG = self.calculate_infogain(X, y, i, t)
                if(IG > IG_max):
                    IG_max = IG
                    best_feature_no = i
                    best_threshold = t

        return best_feature_no, best_threshold
    
    def fit(self, X, y):
        """
        grow the decision tree on the training data
        args
            X  : ndarray of shape (n_samples, n_features), the training input samples
            y  : ndarray of shape (n_samples,), the target values
        """
        
        start_time = time.time()
        self.root = self.growTree(X, y, 0)
        print(f'Time taken to build tree: {time.time() - start_time} seconds')
        return self
        
    def predict(self, X):
        """
        predict the class labels for the input samples
        
        arg
            X  : ndarray of shape (n_samples, n_features), the input samples
        
        return
            list of predicted output
        """
        
        return np.asarray([self._predict(inputs) for inputs in X])
            
    def _predict(self, x):
        node = self.root
        y = 0
        while node:
            if(node.leaf_value is not None):
                y = node.leaf_value
                break

            elif(x[node.feature] <= node.threshold):
                node = node.left
            else:
                node = node.right                
        return y
    
    def growTree(self, X, y, depth):
        """
        Recursively grow the decision tree
        
        args:
            X     : ndarray of shape (n_samples, n_features), the input samples
            y     : ndarray of shape (n_samples,), the target values
            depth : int, the current depth of the decision tree
            
        return
            Node
        """
        
        y0 = len(y[y==0])
        y1 = len(y[y==1])
        
        # stopping conditions
        if(depth == self.max_depth or len(y) <= self.min_samples_split):
            leaf_node = Node(None, None)
            leaf_node.isLeaf = True
            if(y0 < y1):
                leaf_node.leaf_value = 1
            else:
                leaf_node.leaf_value = 0
            return leaf_node
        
        elif(y0 == 0):
            leaf_node = Node(None, None)
            leaf_node.isLeaf = True
            leaf_node.leaf_value = 1
            return leaf_node
        
        elif(y1 == 0):
            leaf_node = Node(None, None)
            leaf_node.isLeaf = True
            leaf_node.leaf_value = 0
            return leaf_node
                        
        # recursive tree building
        else:
            
            # selecting best feature number and threshold
            if(self.criterion == 'entropy'):
                best_feature = self.bestFeature_a(X, y)
                threshold = np.median(X[:,best_feature])
#                 best_feature, threshold = self._find_best_split(X, y)

            else:
#                 best_feature, threshold = self._find_best_split_ginni(X, y)
                best_feature, threshold = self.bestFeature_ginni(X, y)
            
            # spliting the the data set
            X_left = X[np.where(X[:, best_feature] <= threshold)]
            X_right = X[np.where(X[:, best_feature] > threshold)]

            y_left = y[np.where(X[:, best_feature] <= threshold)]
            y_right = y[np.where(X[:, best_feature] > threshold)]

            # if left node became empty -> make leaf
            if(len(y_left) == 0):
                leaf_node = Node(None, None)
                leaf_node.isLeaf = True
                if(len(y_right) > len(y_left)):
                    leaf_node.leaf_value = 1
                else:
                    leaf_node.leaf_value = 0
                return leaf_node
            
            # if right node became empty -> make leaf
            elif(len(y_right) == 0):
                leaf_node = Node(None, None)                
                leaf_node.isLeaf = True
                if(len(y_right) > len(y_left)):
                    leaf_node.leaf_value = 1
                else:
                    leaf_node.leaf_value = 0
                return leaf_node
           
            # grow left and right tree for current node
            else:            
                newNode = Node(best_feature, threshold)
                newNode.left = self.growTree(X_left, y_left, depth+1)
                newNode.right = self.growTree(X_right, y_right, depth+1)
                return newNode
            
    def plot(self):
        """
        Visualize the decision tree using Graphviz.
        """
        dot = graphviz.Digraph()
        self._plot_node(dot, self.root)
        dot.render('decision_tree.pdf', view=True, format='pdf')
    
    def _plot_node(self, dot, node):
        """
        Recursively plot a node and its children.
        """
        if node.isLeaf:
            dot.node(str(id(node)), label=str(node.leaf_value))
        else:
            dot.node(str(id(node)), label=f'X{node.feature} <= {node.threshold}')
            if node.left:
                self._plot_node(dot, node.left)
                dot.edge(str(id(node)), str(id(node.left)), label='True')
            if node.right:
                self._plot_node(dot, node.right)
                dot.edge(str(id(node)), str(id(node.right)), label='False')