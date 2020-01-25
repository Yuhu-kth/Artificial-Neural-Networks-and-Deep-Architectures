import numpy as np 
import math
from math import *
import matplotlib.pyplot as plt

np.random.seed(10)

epoch = 25
etha = 0.001
Alpha = 0.4
Nhidden = 4 # the number of nodes in hidden layer, not sure so far

def Perceptron(X,T,etha,y):
    return -etha*(y - T).dot(X.T)

def Delta_rule(X, T, W, etha):
    """ 
    """
    delta_W = -np.dot(etha,(np.matmul(np.mat(W),np.mat(X))-T))*np.transpose(X)
    return delta_W

def firstW(n,m):
    """initialization of weight matrix
    
    :param n: number of the row of inputs matrix
    :type n: int
    :param m: number of the row of output matrix
    :type m: int
    :return: weight matrix
    :rtype: array
    """
    w = np.random.normal(0,0.5,size=(m,n))
    return w

def bias(x):
    bias = np.ones((1, x.shape[1]))
    x = np.append(x, bias, axis=0)
    return x

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

W = firstW(X.shape[0],T.shape[0])
Wp = W.copy()
yp = np.zeros((T.shape))

print(X.shape)

xx, yy = np.meshgrid(np.arange(-3,3,0.01), np.arange(-2,2,0.01))
xy = np.array((xx.ravel(),yy.ravel()))
grid = bias(xy)

errorDelta = []
errorPerceptron = []
eD = []
eP = []
for i in range(epoch):
    for j in range(X.shape[1]):
        x = X[:,j]
        x = x.reshape(-1,1)
        t = T[:,j]
        yseq = yp[:,j]
    #    print(x,t)
        W += Delta_rule(x,t,W,etha)
        Wp += Perceptron(x,t,etha,yseq)
        yseq = Wp.dot(x)
        yseq = np.where(yseq>=0,1,-1)
        errorDelta.append(-W.dot(x) - t)
        errorPerceptron.append(-yseq-t)
        print(errorPerceptron[-1])
        print(errorDelta[-1])

        Y = W.dot(grid)
        Y = np.where(Y>=0,1,-1)
        Y = Y.reshape(xx.shape)

        ypg = Wp.dot(grid)
        ypg = np.where(ypg>=0,1,-1)
        ypg = ypg.reshape(xx.shape)
        if j ==199 and i>100:
            print("epoch: ",i+1)
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
    eD.append(np.sum(errorDelta))
    eP.append(np.sum(errorPerceptron))
    errorDelta = []
    errorPerceptron = []

x = np.arange(1,len(eD)+1,1)
plt.plot(x,eD,'b' ,label='Delta rule')
plt.plot(x,eP,'r',label='Perceptron')
plt.legend(title='Learning rate')
plt.show()





