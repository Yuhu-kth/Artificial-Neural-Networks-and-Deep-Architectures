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
	output = np.where(x>=0.5,1,-1)
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

def randomData(N, size):
	np.random.seed(10)
	data = np.zeros((N,size))
	for j in range(N):
		data[j]= np.random.choice([-1,1], size)
	return data

def removeDiagonal(w):
	for i in range(w.shape[0]):
		for j in range(w.shape[1]):
			if i==j:
				w[i][j] = 0
	return w

patterns = randomData(300,100)
np.random.seed(10)
print(patterns.shape)

# for i in range(9):
# 	im = patterns[i,:].reshape(32,-1)
# 	print(im.shape)
# 	plt.imshow(im)
# 	plt.show()
errorList = []
errorList2 = []

for i in range(1,patterns.shape[0]):
    for p in np.arange(0.0,0.1,0.1):
        training = patterns[:i,:]
        distorted = noise(training,p)
        W = Weights(training,True)
        W2 = removeDiagonal(W)
        # print(distorted.shape)
        epochs = 1
        X = distorted[0:i]
        error = 0
        #print("p: ",p)
        for e in range(epochs):
            Xnew = update(X,W)
            X = np.copy(Xnew)
            Xnew2 = update(X,W2)
            X2 = np.copy(Xnew2)
        error = 1*(X != patterns[0:i])
        error = sum(sum(error))
        error2 = 1*(X2 != patterns[0:i])
        error2 = sum(sum(error2))
    errorList.append(error)
    errorList2.append(error2)

print(patterns[0])
print(errorList)
        
z = 0
z2 = 0
plt.plot(np.arange(len(errorList))+1,errorList, label= 'Normal network')
plt.plot(np.arange(len(errorList))+1,errorList2, label = 'Non self-connections')

while (0 in errorList):
	errorList.remove(0)
	z+=1

while (0 in errorList2):
	errorList2.remove(0)
	z2+=1	
plt.title("Max number of patterns stored: %i, without self-connections: %i " %(z,z2))
plt.xlabel("number of patterns added")
plt.ylabel("Error")
plt.legend()
plt.show()
