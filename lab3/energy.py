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


data = np.genfromtxt('pict.dat',delimiter =',').reshape(11,-1)

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
W = 0.5 * (W+W.T)
#W = np.random.randn(1024,1024)
print(W.shape)
trainingPatterns=data[:3,:]
testingPatterns=data[3:,:]
#energy at different attractors
for i in range(len(trainingPatterns)):
    x=trainingPatterns[i]
    print("The energy at attractor p{} is {}".format(i+1,Energy(x,W)))

#energy at the points of distorted patterns
for i in range(len(testingPatterns)):
    x=testingPatterns[i]
    print("The energy at the points of disorted pattern p{} is {}".format(i+4,Energy(x,W)))

epochs = 100
X = distorted[0]
energyList=[]
for e in range(epochs):
	# Xnew = update(X,W)
	Xnew = updateRandom(X,W)
	X = np.copy(Xnew)
	energyList.append(Energy(X,W))

plt.plot(range(epochs),energyList)
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.title("Energy of distorted pattern p1 change with the iteration(random update) and symmetric W")
plt.show()


