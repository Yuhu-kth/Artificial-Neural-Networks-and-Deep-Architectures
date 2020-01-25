"""Classification of samples that are not linearly separable"""
import numpy as np 
import math
from math import *
import matplotlib.pyplot as plt
import random 
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
    
    # mA = [1.0, 0.5]
    # mB = [-1.0, 0.0]
    mA = [1.0,0.3]
    mB = [0.0,-0.1]
    sigmaA = 0.2
    sigmaB = 0.3

    classA = np.zeros((2,n))
    classB = np.zeros((2,n))
    #TODO check this line below
    classA[0] = np.hstack((np.random.randn(1,round(0.5*n)) * sigmaA - mA[0], np.random.randn(1,round(0.5*n)) * sigmaA + mA[0]))
    classA[1][:] = np.random.randn(1,n) * sigmaA + mA[1]
    classB[0][:] = np.random.randn(1,n) * sigmaB + mB[0]
    classB[1][:] = np.random.randn(1,n) * sigmaB + mB[1]

    #remove data samples
    choice = 1
    if choice == 1:
        new_classA = np.zeros((2,75))
        new_classB = np.zeros((2,75))
        #1.random 25% from each class
        np.random.shuffle(classA)
        new_classA[0] = classA[0][25:]
        new_classA[1] = classA[1][25:]
        np.random.shuffle(classB)
        new_classB[0] = classB[0][25:]
        new_classB[1] = classB[1][25:]
        T = np.ones((1,n*2-50))
        T[0][n:] += -2
    elif choice == 2:
        #2.random 50% from classA
        new_classA = np.zeros((2,50))
        np.random.shuffle(classA)
        new_classA[0] = classA[0][50:]
        new_classA[1] = classA[1][50:]
        T = np.ones((1,n*2-50))
        T[0][n:] += -2

    elif choice == 3:
        #3.random 50% from classB
        new_classB = np.zeros((2,50))
        np.random.shuffle(classB)
        new_classB[0] = classB[0][50:]
        new_classB[1] = classB[1][50:]
        T = np.ones((1,n*2-50))
        T[0][n:] += -2
    elif choice == None:
        pass
    # elif choice == 4:
    #     #4.20% from classA(1,:)<0, 80 from classA(1,:)>0
    #     # subsetA = np.zeros
    #     # for i in range(len(classA[0][:])):
    #     #     if classA[0][i] < 0 :
    #     #         subsetA.append(classA[0][i])
    



    # X = np.concatenate((classA,classB),axis=1)
    X = np.concatenate((new_classA,new_classB),axis=1)
    X = bias(X)

    W = firstW(X.shape[0],T.shape[0])
    Wp = W.copy()
    yp = np.zeros((T.shape))


    return new_classA, new_classB, X, T, W, Wp, yp

def Perceptron(X,T,etha,y):
	return -etha*(y - T).dot(X.T)

def Delta_rule(X, T, W, etha):
    """ 
    """
    delta_W = -np.dot(etha,(np.matmul(np.mat(W),np.mat(X))-T))*np.transpose(X)
    return delta_W

def bias(x):
	bias = np.ones((1, x.shape[1]))
	x = np.append(x, bias, axis=0)
	return x

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

def phi(X):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = 2/(1+math.exp(-X[i][j])) - 1
    return X

def learning():
    classA, classB, X, T, W, Wp, yp = _init_()

    for i in range(epoch):
        W += Delta_rule(X, T, W, etha)
        Wp += Perceptron(X,T,etha,yp)
        yp = Wp.dot(X)
        yp = np.where(yp>=0,1,-1)
        # print(W)
        # plot the decision boundary: Wx=0
        # plt.plot(X,WX)
        
        xx, yy = np.meshgrid(np.arange(-3,3,0.01), np.arange(-2,2,0.01))
        xy = np.array((xx.ravel(),yy.ravel()))
        grid = bias(xy)

        Y = W.dot(grid)
        # Y = W.dot(xy)
        Y = np.where(Y>=0,1,-1)
        Y = Y.reshape(xx.shape)

        ypg = Wp.dot(grid)
        # ypg = Wp.dot(xy)
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
if __name__ == "__main__":
    learning()



