Overview
svm_binary.py contains the Trainer class which is used to fit and predict binary classification problems using support vector machines (SVMs). The Trainer class has the following methods:

fit(self, train_data_path:str): Fit the SVM model to the binary training data X and y using a specified regularization parameter C, kernel function kernel, and maximum number of iterations max_iter. This method returns the trained SVM model.
predict(self, test_data_path:str): Predict the binary class labels for the input data X using the trained SVM model model. This method returns a 1D array of predicted binary class labels.
Additionally, there are two helper functions fit_multi() and predict_multi() which are used by the svm_multiclass.py script to train and predict on the multi-class classification problems by combining binary classifiers.

svm_multiclass contains two classifier, One-vs-One and One-vs-All. Each of them have two functions fit() and predict() having same signature.

kernel.py contains the function to calculate kernel matrix.
supported kernel are:
Linear
Polynomial
RBF
Sigmoid
Laplacian


Dependencies
This code requires the following dependencies:

numpy (for numerical operations)
matplotlib (for visualization)
qpsolver (for solving dual problem)
pandas (for reading data frame)

Usage
To use the Trainer class for binary classification, import the Trainer class from svm_binary.py and create an instance of it. Then, call the fit() method to fit the SVM model to the training data and the predict() method to predict the binary class labels for new data.