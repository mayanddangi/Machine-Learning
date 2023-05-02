import numpy as np

# Do not change function signatures
#
# input:
#   X is the input matrix of size n_samples x n_features.
#   pass the parameters of the kernel function via kwargs.
# output:
#   Kernel matrix of size n_samples x n_samples 
#   K[i][j] = f(X[i], X[j]) for kernel function f()

def linear(X: np.ndarray, **kwargs)-> np.ndarray:
    Z = X
    for key, value in kwargs.items():
        if(key=='Z'):
            Z = value
    assert X.ndim == 2
    kernel_matrix = X @ Z.T
    return kernel_matrix

'''
    K(x,y) = (zeta + gamma X^T X)^Q
'''
def polynomial(X:np.ndarray,**kwargs)-> np.ndarray:
    Q = 1
    gamma = 1
    zeta = 1
    Z = X
    for key, value in kwargs.items():
        if(key=='degree'):
            Q = value
        if(key=='gamma'):
            gamma = value
        if(key=='zeta'):
            zeta = value
        if(key == 'Z'):
            Z = value
    kernel_matrix = np.power(zeta + gamma * X @ Z.T, Q)
    return kernel_matrix

def rbf(X:np.ndarray,**kwargs)-> np.ndarray:
    Z = X
    gamma = 1
    for key, value in kwargs.items():
        if(key=='Z'):
            Z = value
        if(key=='gamma'):
            gamma = value
    temp = np.sum(X**2, axis=1, keepdims = True) -2*np.matmul(X, Z.T) + np.sum(Z**2, axis=1, keepdims = True).T
    return np.exp(-gamma*temp)

def sigmoid(X:np.ndarray,**kwargs)-> np.ndarray:
    offset = 0
    gamma = 1
    Z = X
    for key, value in kwargs.items():
        if(key=='offset'):
            offset = value
        if(key=='gamma'):
            gamma = value
        if(key=='Z'):
            Z = value
    kernel_matrix = np.tanh(gamma * np.dot(X, Z.T) + offset)
    return kernel_matrix

def laplacian(X:np.ndarray,**kwargs)-> np.ndarray:
    Z = X
    gamma = 1
    for key, value in kwargs.items():
        if(key=='Z'):
            Z = value
        if(key=='gamma'):
            gamma = value
            
    X_len = X.shape[0]
    Z_len = Z.shape[0]
    
    kernel_matrix = np.zeros((X_len, Z_len))
    for i in range(0, X_len):
        for j in range(0, Z_len):
            kernel_matrix[i, j] = np.exp(-1*gamma*np.sum(np.abs(X[i] - Z[j])))
    return kernel_matrix

