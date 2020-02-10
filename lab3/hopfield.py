import numpy as np
from itertools import product

def Weights(x, scaling):
	n=len(patterns[0])
	W = (x.T).dot(x)
	if scaling == True:
		W = W/n	
	return W


def update(x,w):
	X = np.dot(x,w)
	for i in range(len(X)):
		X[i]=sign(X[i])
	return X


def sign(x):
	output = np.where(x>=0,1,-1)
	return output

def bias(x):
 	bias = np.ones((x.shape[0],1))
 	x = np.append(x, bias, axis=1)
 	return x

patterns = np.array([[-1,-1,1,-1,1,-1,-1,1],[-1,-1,-1,-1,-1,1,-1,-1],[-1,1,1,-1,-1,1,-1,1]], dtype = float)
patterns = bias(patterns)


W = Weights(patterns,True)
print(W.shape)

distorted = np.array([[1,-1,1,-1,1,-1,-1,1],[1,1,-1,-1,-1,1,-1,-1],[1,1,1,-1,1,1,-1,1]], dtype = float)

epochs = 10
X = np.copy(bias(distorted))
for e in range(epochs):
	Xnew = update(X,W)
	if (X == patterns).all():
		print("Stop crietria, itratation: ",e)
		break
	X = np.copy(Xnew)

print(Xnew==patterns)
print(Xnew)

p = product([-1,1],repeat = 8)

#print(len(list(p)))
attr = np.zeros((2**8,8))
i = 0
for ps in list(p):
	attr[i] = ps
	i +=1
print(i)
print(attr.shape)

attr = bias(attr)
for e in range(epochs):
	Xnew2 = update(attr,W)
	attr = np.copy(Xnew2)
print("Number of attractors: ", len(np.unique(attr,axis=0)))
print(np.unique(attr,axis=0))
