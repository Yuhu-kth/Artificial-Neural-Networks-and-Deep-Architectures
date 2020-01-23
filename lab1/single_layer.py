import numpy as np 
import math
from math import *
import matplotlib.pyplot as plt

epoch = 20
etha = 0.001
Alpha = 0.4
Nhidden = 4 # the number of nodes in hidden layer, not sure so far
def _init_():
    """This function generates data, targets and weight matrix
    :return: X,the matrix of input patterns
    :rtype: class 'numpy.ndarray'
    :return: T, targets
    :rtype: class 'numpy.ndarray'
    :return: W, weight matrix
    :rtype: class 'numpy.ndarray'
    """
    n = 100
    
    mA = [1.0, 0.5]
    mB = [-1.0, 0.0]
    sigma = 0.2

    classA = np.zeros((2,n))
    classB = np.zeros((2,n))
    
    classA[0][:] = np.random.randn(1,n) * sigma + mA[0]
    classA[1][:] = np.random.randn(1,n) * sigma + mA[1]
    classB[0][:] = np.random.randn(1,n) * sigma + mB[0]
    classB[1][:] = np.random.randn(1,n) * sigma + mB[1]

    X = np.concatenate((classA,classB),axis=1)
    X = bias(X)
    T = np.ones((1,n*2))
    T[0][n:] += -2
    W = firstW(X,T)
    Wp = W.copy()
    yp = np.zeros((T.shape))


    return classA, classB, X, T, W, Wp, yp

def Perceptron(X,T,etha,y):
	return -etha*(y - T).dot(X.T)

def Delta_rule(X, T, W, etha):
    """ 
    """
    delta_W = -np.dot(etha,(np.matmul(np.mat(W),np.mat(X))-T))*np.transpose(X)
    return delta_W

def single_layer_perceptron(X, T , y, etha):
    delta_W = np.dot(etha,(T-y))*np.transpose(X)
    return delta_W

def bias(x):
	bias = np.ones((1, x.shape[1]))
	x = np.append(x, bias, axis=0)
	return x

def firstW(X,T):
    n = X.shape[0] # should be the row of inputs matrix
    m = T.shape[0] # should be the row of Tagets matrix
    w = np.random.normal(0,0.5,size=(m,n))
    return w

# def two_layer_perceptron(x,nHidden):
#     phi = 2/(1+math.exp(-x)) - 1
#     phi_prime = 0.5*(1+phi)*(1-phi)
#     pass
def phi(x):
    phi = 2/(1+math.exp(-x)) - 1
    phi_prime = 0.5*(1+phi)*(1-phi)
    return phi,phi_prime

# implementation of 2.3.1-2.3.3 following the instructions 
def backprop(Nhidden):
    # the forward pass
    X, T = _init_()
    W = firstW(X,nHidden)
    v = firstW(nHidden,1)
    hin = W*X
    hout = phi(hin).phi
    oin = v*hout #TODO: not sure what is v!
    out = phi(oin).phi
    # the backward pass
    delta_o = 0.5*np.multiply((out-T),np.multiply((1+out),(1-out)))
    delta_h = 0.5*np.multiply(v.T * delta_o, np.multiply((1+hout),(1-hout))) #TODO: not sure what is V_prime
    delta_h = delta_h(1:Nhidden,:)
    #Weight update TODO: check v !
    dw = np.multiply(np.multiply(dw, alpha)- (delta_h*X), (1-Alpha))
    dv = np.multiply(np.multiply(dv, alpha)- (delta_o*hout.T), (1-Alpha))
    W = W + np.multiply(dw,etha)
    V = v + np.multiply(dv,etha)
    return W, V

def learning():
    classA, classB, X, T, W, Wp, yp = _init_()

    for i in range(epoch):
        W += Delta_rule(X, T, W, etha)
        Wp += Perceptron(X,T,etha,yp)
        yp = Wp.dot(X)
        yp = np.where(yp>=0,1,-1)
        print(W)
        # plot the decision boundary: Wx=0
        # plt.plot(X,WX)
        
        xx, yy = np.meshgrid(np.arange(-3,3,0.01), np.arange(-2,2,0.01))
        xy = np.array((xx.ravel(),yy.ravel()))
        grid = bias(xy)

        Y = W.dot(grid)
        Y = np.where(Y>=0,1,-1)
        Y = Y.reshape(xx.shape)


        ypg = Wp.dot(grid)
        ypg = np.where(ypg>=0,1,-1)
        ypg = ypg.reshape(xx.shape)

        plt.figure()
        plt.subplot(121)
        plt.contourf(xx,yy,Y,alpha = 0.4)
        plt.scatter(classA[0][:],classA[1][:])
        plt.scatter(classB[0][:], classB[1][:])
        plt.title("Delta Learning")
        plt.subplot(122)
        plt.contourf(xx,yy,ypg,alpha = 0.4)
        plt.scatter(classA[0][:],classA[1][:])
        plt.scatter(classB[0][:], classB[1][:])
        plt.title("Perceptron Learning")
        plt.show()

def sequential_learning():
    for i in range(epoch):
        backprop(Nhidden)
    pass


if __name__ == "__main__":
    # single_layer_perceptron()
    learning()

