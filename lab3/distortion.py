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


def Energy(x,w):
	dim = x.size
	energy = 0
	for i in range(dim):
		for j in range(dim):
			energy += w[i,j]*x[i]*x[j]
	return -energy


def updateRandom(x,w):
	N = len(x)
	Xnew = np.copy(x)
	for num in range(N):
		index = np.random.randint(0,N)
		temp = w[index,:].dot(x)
		Xnew[index] = sign(temp)
	return Xnew

def noise(x,percent):
	p = np.copy(x)
	l = int(percent * x.shape[1])
	ind = np.arange(x.shape[1])
	indDis = np.random.choice(ind,l)
	for i in range(p.shape[0]):
		p[i,indDis] *= -1
	return p





data = np.genfromtxt('pict.dat',delimiter =',').reshape(11,-1)

patterns = data[:9,:]

print(patterns.shape)
"""
for i in range(3):
	im = patterns[i,:].reshape(32,-1)
	print(im.shape)
	plt.imshow(im)
	plt.show()

"""
for p in np.arange(0.1,1.1,0.1):
	training = patterns[:3,:]
	distorted = noise(training,p)


	print(training.shape)
	W = Weights(training,True)
	print(W.shape)

	epochs = 100
	X = distorted[2]
	energyList=[]
	for e in range(epochs):
		# Xnew = update(X,W)
		Xnew = update(X,W)
		X = np.copy(Xnew)

	fig = plt.figure()
	a = fig.add_subplot(1,3,1)
	plt.imshow(patterns[2,:].reshape(32,-1))
	a.set_title('Training Pattern P3')
	a = fig.add_subplot(1,3,2)
	plt.imshow(distorted[2,:].reshape(32,-1))
	a.set_title('Pattern of P3 with %.2f of degradation'%p)
	a = fig.add_subplot(1,3,3)
	plt.imshow(X.reshape(32,-1))
	a.set_title('Completed pattern')
	plt.show()

