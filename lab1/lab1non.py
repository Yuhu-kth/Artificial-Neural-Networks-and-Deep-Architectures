import numpy as np 
import math
from math import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    delta_o = 0.5 * (out-T) * ((1 + out) * (1 - out))
    delta_h = 0.5 * (v.T.dot(delta_o)) * ((1 + hout) * (1 - hout))
    delta_h = delta_h[0:Nhidden,:]

    #Weight update 
    dw = (dw * Alpha) - delta_h.dot(X.T) * (1-Alpha)
    dv = (dv * Alpha) - delta_o.dot(hout.T) * (1-Alpha)
    w = w + dw*etha
    v = v + dv*etha
    return w, v, dw, dv

"""
    print(dw.shape)
    print(dv.shape)
    print("W:",w)
    print("V:",v)
    print("dw:",dw)
    print("dv:",dv)"""

def phi(X):
    return 2 /(1+np.exp(-X))-1

epoch = 500
etha = 0.01
Alpha = 0.9
Nhidden = 5 # the number of nodes in hidden layer, not sure so far
#Nhidden = np.arange(1,20,1)
np.random.seed(1)
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

pA = 0
pB = 50

trainingA = np.zeros((2,100-pA))
trainingB = np.zeros((2,100-pB))
testA = np.zeros((2,pA))
testB = np.zeros((2,pB))
#1.random 25% from each class
np.random.shuffle(classA)

a = pA//2
aa = pA - pA//2
aaa = pA + pA//2 
trainingA[0][:a] = classA[0][:aa]
trainingA[1][:a] = classA[1][:aa]
trainingA[0][a:] = classA[0][aaa:]
trainingA[1][a:] = classA[1][aaa:]

testA[0] = classA[0][aa:aaa]
testA[1] = classA[1][aa:aaa]

np.random.shuffle(classB)
trainingB[0] = classB[0][pB:]
trainingB[1] = classB[1][pB:]
testB[0] = classB[0][:pB]
testB[1] = classB[1][:pB]
Ttrain = np.ones((1,n*2-pA-pB))
Ttrain[0][n-pA:] += -2

Ttest = np.ones((1,pA+pB))
Ttest[0][pA:] += -2

Xtrain = np.concatenate((trainingA,trainingB),axis=1)
Xtrain = bias(Xtrain)
Xtest = np.concatenate((testA,testB),axis=1)
Xtest = bias(Xtest)

X = np.concatenate((classA,classB),axis=1)
X = bias(X)

T = np.ones((1,n*2))
T[0][n:] += -2
#Initialization
dw = np.zeros((Nhidden,X.shape[0]))
dv = np.zeros((1,Nhidden+1))

w = np.random.randn(Nhidden,X.shape[0])
v = np.random.randn(1,Nhidden+1)

xx, yy = np.meshgrid(np.arange(-2,2,0.01), np.arange(-2,2,0.01))
xy = np.array((xx.ravel(),yy.ravel()))
grid = bias(xy)

etrain = []
etest = []
#for h in Nhidden:
for i in range(epoch):
    w, v, dw, dv = backprop(Nhidden,Xtrain,Ttrain,w,v,dw,dv)

    ytrain = phi(v.dot(bias(phi(w.dot(Xtrain)))))
    error = mean_squared_error(Ttrain,ytrain)
    ytrain = np.where(ytrain>=0,1,-1)
    mis = np.where(ytrain == Ttrain,0,1)
    etrain.append(error)

    ytest = phi(v.dot(bias(phi(w.dot(Xtest)))))
    error2 = mean_squared_error(Ttest,ytest)
    ytest = np.where(ytest>=0,1,-1)
    mis2 = np.where(ytest == Ttest,0,1)
    etest.append(error2)

print("Hidden: ",Nhidden)
print("MSE training: ",error)
print("Misclassified ratio training: ",np.sum(mis)/(Xtrain.shape[1]*2))
print("MSE test: ",error2)
print("Misclassified ratio test: ",np.sum(mis2)/(Xtest.shape[1]*2))

x = np.arange(1,epoch+1,1)
plt.plot(x,etrain,label='Train error')
plt.plot(x,etest,label='Test error')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()
    #if i % 5 == 0:
"""Y = phi(v.dot(bias(phi(w.dot(grid)))))
Y = np.where(Y>=0,1,-1)
Y = Y.reshape(xx.shape)
plt.scatter(trainingA[0][:],trainingA[1][:], facecolor = 'blue', marker = '+', label = "Training set class A")
plt.scatter(trainingB[0][:], trainingB[1][:], facecolor = 'red', marker = '+', label = "Training set class B")
plt.scatter(testA[0][:],testA[1][:], facecolor = 'blue', label = "Validation set class A")
plt.scatter(testB[0][:], testB[1][:], facecolor = 'red', label = "Validation set class B")
plt.contourf(xx,yy,Y,alpha = 0.4)
plt.legend()
plt.title("50% training for class B")
plt.show()
"""