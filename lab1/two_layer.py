import numpy as np 
import math
from math import *
import matplotlib.pyplot as plt

def bias(x):
    bias = np.zeros((1,x.shape[1]))
    x = np.append(x, bias, axis=0)
    return x

def backprop(Nhidden,X,T,w,v,dw,dv):
    # the forward pass
    hin = w.dot(X)
    print(hin.shape)
    hout = phi(hin)
    print(hout.shape)
    hout = bias(hout)

    oin = v.dot(hout) 
    out = phi(oin)

    # the backward pass
    delta_o = 0.5*np.multiply((out-T),np.multiply((1+out),(1-out)))
    delta_h = 0.5*np.multiply(v.T * delta_o, np.multiply((1+hout),(1-hout)))
    delta_h = delta_h[0:Nhidden,:]


    #Weight update 
    dw = np.dot(dw, Alpha)- np.dot(delta_h,np.transpose(X))
    dv = np.dot(dv, Alpha)- np.dot(delta_o,np.transpose(hout))
    W = w + np.multiply(dw,etha)
    V = v + np.multiply(dv,etha)
    print("W:",W)
    print("V:",V)
    return W, V

def phi(X):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = 2/(1+math.exp(-X[i][j])) - 1
    return X

epoch = 50
etha = 0.001
Alpha = 0.4
Nhidden = 4 # the number of nodes in hidden layer, not sure so far

np.random.seed(13)
n = 100
    
mA = [1.0, 0.3]
mB = [0.0, -0.1]
sigmaA = 0.2
sigmaB = 0.3

classA = np.zeros((2,n))
classB = np.zeros((2,n))

classA[0,:n//2] = np.random.randn(1,n//2) * sigmaA - mA[0]
classA[0,n//2:] = np.random.randn(1,n//2) * sigmaA + mA[0]
classA[1][:] = np.random.randn(1,n) * sigmaA + mA[1]
classB[0][:] = np.random.randn(1,n) * sigmaB + mB[0]
classB[1][:] = np.random.randn(1,n) * sigmaB + mB[1]

X = np.concatenate((classA,classB),axis=1)
X = bias(X)

T = np.ones((1,n*2))
T[0][n:] += -2

#Initialization
dw = np.zeros((Nhidden,X.shape[0])) #TODO check the dimensions of matrix
dv = np.zeros((1,Nhidden+1))

w = np.random.randn(Nhidden,X.shape[0])
v = np.random.randn(1,Nhidden+1)
for i in range(epoch):
    W, V = backprop(Nhidden,X,T,w,v,dw,dv)
    w = np.copy(W)
    v = np.copy(V)

xx, yy = np.meshgrid(np.arange(-2,2,0.01), np.arange(-2,2,0.01))
xy = np.array((xx.ravel(),yy.ravel()))
grid = bias(xy)

Y = phi(v.dot(bias(phi(w.dot(grid)))))
Y = np.where(Y>=0,1,-1)
Y = Y.reshape(xx.shape)

plt.scatter(classA[0][:],classA[1][:])
plt.scatter(classB[0][:], classB[1][:])
plt.contourf(xx,yy,Y,alpha = 0.4)
plt.show()
