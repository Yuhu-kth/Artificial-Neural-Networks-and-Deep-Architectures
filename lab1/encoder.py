import numpy as np 
import math
from math import *
import matplotlib.pyplot as plt

def bias(x):
    bias = np.ones((1,x.shape[1]))
    x = np.append(x, bias, axis=0)
    return x

def backprop(Nhidden,X,T,w,v,dw,dv):
    # the forward pass
    hin = w.dot(X)
    hout = phi(hin)
    hout = bias(hout)

    oin = v.dot(hout) 
    out = phi(oin)

    # the backward pass
    delta_o = 0.5*np.multiply((out-T),np.multiply((1+out),(1-out)))
    delta_h = 0.5*np.multiply(np.dot(v.T,delta_o), np.multiply((1+hout),(1-hout)))
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

epoch = 300
etha = 0.03
Alpha = 0.4
Nhidden = 3 # the number of nodes in hidden layer, not sure so far

X = np.eye(8)*2-1
X = bias(X)
T = np.eye(8)*2-1

#Initialization
dw = np.zeros((Nhidden,X.shape[0])) #TODO check the dimensions of matrix
dv = np.zeros((T.shape[0],Nhidden+1))

w = np.random.randn(Nhidden,X.shape[0]) # 3X9
v = np.random.randn(T.shape[0],Nhidden+1) # 8X4
for i in range(epoch):
    W, V = backprop(Nhidden,X,T,w,v,dw,dv)
    w = np.copy(W)
    v = np.copy(V)

# xx, yy = np.meshgrid(np.arange(-2,2,0.01), np.arange(-2,2,0.01))
# xy = np.array((xx.ravel(),yy.ravel()))
# grid = bias(xy)


Y = w.dot(X)
Y = phi(Y)
Y = bias(Y)
Y = v.dot(Y)
Y = phi(Y)

Y = np.where(Y>=0,1,-1)
print("Y:",Y)
print("T:",T)
# Y = Y.reshape(xx.shape)

# plt.scatter(X)
# plt.contourf(xx,yy,Y,alpha = 0.4)
# plt.show()
