import numpy as np 
<<<<<<< HEAD
import math
from math import *
import matplotlib.pyplot as plt
#implementation of Delta rule
def Delta_rule(X, T, W, etha):
    """ :
=======


#implementation of Delta rule
def Delta_rule(X, T, W):
    """ epoch:
        eta:
        delta_W:
>>>>>>> 6d4cb105b986a0f835ac0743d39389095dd91a08
    """
    delta_W = -etha*(W*X-T)*X.transpose
    return delta_W

def bias(x):
	bias = np.ones((1,x.shape[1]))
	x = np.append(x, bias, axis=0)
	return x

def firstW(X,T):
    n = X.shape[0]
    m = T.shape[0]
    w = np.random.randn(0,(m,n))
    return w
    
def _init_():
    n = 100
    
    mA = [1.0, 0.5]
    mB = [-1.0, 0.0]
    sigma = 0.5

    classA = np.zeros((2,n))
    classB = np.zeros((2,n))
    
    classA[0][:] = np.random.randn(1,n) * sigma + mA[0]
    classA[1][:] = np.random.randn(1,n) * sigma + mA[1]
    classB[0][:] = np.random.randn(1,n) * sigma + mB[0]
    classB[1][:] = np.random.randn(1,n) * sigma + mB[1]

    X = np.concatenate((classA,classB),axis=1)
    X = bias(X)
    T = np.ones((n*2))
    T[n:] += -2
    W = firstW(X,T)
    # plt.scatter(classA[0][:],classA[1][:])
    # plt.scatter(classB[0][:], classB[1][:])
    # plt.show()
    return X, T, W

def two_layer_perceptron(x):
    e = math.e
    phi = 2/(1+e.math.exp(-x)) - 1
    gradientPhi = 0.5*(1+phi)*(1-phi)
    pass

def single_layer_perceptron():
    X,T,W = _init_()
    epoch = 20
<<<<<<< HEAD
    etha = 0.01

    for i in range(epoch):
        new_delta_W = Delta_rule(X, T, W, etha)
        delta_W += new_delta_W
        # plot the decision boundary: Wx=0
        # plt.plot(X,WX)
    pass
# generate_data()
=======
    eta = 0.5
    delta_W = -eta*(W*X-T)*X.transpose
    return delta_W

def bias(x):
	bias = np.ones((1,x.shape[1]))
	x = np.append(x, bias, axis=0)
	return x

def firstW(x,t):
	n = x.shape[0]
	m = t.shape[0]
	w = np.random.randn(0,(m,n))
	return w

etha = 0.001
n = 100
mA = [1.0, 0.5]
mB = [-1.0, 0.0]
sigma = 0.5

classA = np.zeros((2,n))
classB = np.zeros((2,n))

classA[0][:] = np.random.randn(1,n) * sigma + mA[0]
classA[1][:] = np.random.randn(1,n) * sigma + mA[1]
classB[0][:] = np.random.randn(1,n) * sigma + mB[0]
classB[1][:] = np.random.randn(1,n) * sigma + mB[1]

X = np.concatenate((classA,classB), axis=1)

t = np.ones((n*2))
t[n:] += -2

X = bias(X)
#perceptron
W = firstW(X,t)

epoch = 20

plt.scatter(classA[0][:],classA[1][:])
plt.scatter(classB[0][:], classB[1][:])
plt.show()
>>>>>>> 6d4cb105b986a0f835ac0743d39389095dd91a08
