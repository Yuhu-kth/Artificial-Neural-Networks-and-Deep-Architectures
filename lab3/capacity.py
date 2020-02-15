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

# for i in range(9):
# 	im = patterns[i,:].reshape(32,-1)
# 	print(im.shape)
# 	plt.imshow(im)
# 	plt.show()
errorList = []
for i in range(1,9):
    for p in np.arange(0.1,0.2,0.1):
        training = patterns[:i,:]
        distorted = noise(training,p)
        W = Weights(training,True)
        # print(distorted.shape)
        epochs = 1
        X = distorted[0]
        energyList=[]
        error = 0
        print("p: ",p)
        for e in range(epochs):
            Xnew = update(X,W)
            X = np.copy(Xnew)
        error = 1*(X != patterns[0])
        error = sum(error)
    errorList.append(error)
    print(errorList)
        

plt.plot(np.arange(len(errorList)),errorList)
plt.title("Errors in p1 while increasing the number of training patterns")
plt.xlabel("number of patterns added")
plt.ylabel("Error")
plt.show()
