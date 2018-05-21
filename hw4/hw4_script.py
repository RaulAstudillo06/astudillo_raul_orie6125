import numpy as np
import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot, squared_norm
from scipy.misc import comb, logsumexp 
from sklearn.linear_model.logistic import _multinomial_grad_hess

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Get data
mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64')
y = mnist.target

# Split data into traning and test datasets
train_samples = 30000
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_samples, test_size=10000, random_state=1)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Y_train = np.zeros((len(y_train), 10))
for i,j in enumerate(y_train):
    Y_train[i, int(j)] = 1

Y_test = np.zeros((len(y_test), 10))
for i,j in enumerate(y_test):
    Y_test[i, int(j)] = 1
    
# Loss function, gradient and hessian
def loss_function(X,Y, w, alpha=0):
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    w = w.reshape(n_classes, -1)
    fit_intercept = w.size == (n_classes * (n_features + 1))
    old_w = w.copy()
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(Y * p).sum()
    loss += 0.5 * alpha * squared_norm(w)
    p = np.exp(p, p)
    return loss, p, w

def grad_loss(X, Y, w, alpha=0):
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = (w.size == n_classes * (n_features + 1))
    grad = np.zeros((n_classes, n_features + bool(fit_intercept)),
                    dtype=X.dtype)
    loss, p, w = loss_function(X, Y, w,alpha)
    diff = (p - Y)
    grad[:, :n_features] = safe_sparse_dot(diff.T, X)
    grad[:, :n_features] += alpha * w
    if fit_intercept:
        grad[:, -1] = diff.sum(axis=0)
    return loss, grad.ravel(), p

def hessian_loss(X, Y, w, alpha=0):
    n_features = X.shape[1]
    n_classes = Y.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))

    loss, grad, p = grad_loss(X, Y, w,alpha)
    def hessp(v):
        v = v.reshape(n_classes, -1)
        if fit_intercept:
            inter_terms = v[:, -1]
            v = v[:, :-1]
        else:
            inter_terms = 0
        r_yhat = safe_sparse_dot(X, v.T)
        r_yhat += inter_terms
        r_yhat += (-p * r_yhat).sum(axis=1)[:, np.newaxis]
        r_yhat *= p
        hessProd = np.zeros((n_classes, n_features + bool(fit_intercept)))
        hessProd[:, :n_features] = safe_sparse_dot(r_yhat.T, X)
        hessProd[:, :n_features] += v * alpha
        if fit_intercept:
            hessProd[:, -1] = r_yhat.sum(axis=0)
        return hessProd.ravel()
    w_copy = w.copy()
    w_copy[-1] = 0.0
    return grad, hessp, w_copy

fit_intercept = True

def sgd(momentum=0.9, lr=0.01, batch_size=1001, alpha=0.1, maxepoch=50, eps=1e-8):
    """
    Real-time forward-mode algorithm using stochastic gradient descent with constant learning 
    rate. Observe that you should only find the optimal learning rate (lr), and 
    penalty parameter (alpha). 
    
    We use the SGD with momentum, which is defined here: 
    http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/
    """
    pass
    