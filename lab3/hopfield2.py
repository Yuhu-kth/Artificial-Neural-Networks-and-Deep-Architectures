import numpy as np
from itertools import product
from matplotlib import pyplot as plt

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

def updateRandom(x,w):
	N = len(x)
	Xnew = np.copy(x)
	for num in range(N):
		index = np.random.randint(0,N)
		temp = w[index,:].dot(x)
		Xnew[index] = sign(temp)
	return Xnew

data = np.genfromtxt('pict.dat.txt',delimiter =',').reshape(11,-1)

patterns = data[:9,:]
distorted = data[9:,:]

print(patterns.shape)
print(distorted.shape)
"""
for i in range(3):
	im = patterns[i,:].reshape(32,-1)
	print(im.shape)
	plt.imshow(im)
	plt.show()

"""
W = Weights(patterns[:3,:],True)
print(W.shape)


epochs = 100
X = distorted[0]
for e in range(epochs):
	Xnew = update(X,W)
	X = np.copy(Xnew)

fig = plt.figure()
a = fig.add_subplot(1,3,1)
plt.imshow(patterns[0,:].reshape(32,-1))
a.set_title('Training Pattern P1')
a = fig.add_subplot(1,3,2)
plt.imshow(distorted[0,:].reshape(32,-1))
a.set_title('Degradated Pattern of P1')
a = fig.add_subplot(1,3,3)
plt.imshow(X.reshape(32,-1))
a.set_title('Completed pattern')
plt.show()

epochs = 100
X = distorted[1]
for e in range(epochs):
	# Xnew = update(X,W)
	Xnew = updateRandom(X,W)
	X = np.copy(Xnew)

fig = plt.figure()
a = fig.add_subplot(1,4,1)
plt.imshow(patterns[1,:].reshape(32,-1))
a.set_title('Training Pattern P2')
a = fig.add_subplot(1,4,2)
plt.imshow(patterns[2,:].reshape(32,-1))
a.set_title('Training Pattern P3')
a = fig.add_subplot(1,4,3)
plt.imshow(distorted[1,:].reshape(32,-1))
a.set_title('Degradated Patterns of P2 and P3')
a = fig.add_subplot(1,4,4)
plt.imshow(X.reshape(32,-1))
a.set_title('Completed pattern')
plt.show()
