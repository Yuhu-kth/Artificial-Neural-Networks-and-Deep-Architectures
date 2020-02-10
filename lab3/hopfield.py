import numpy as np

def Weights(x, units):
	W = (x.T).dot(x)
	#W = W/units
	return W

def weightMatrix(patterns,scaling):
	n=len(patterns[0])
	p=len(patterns)
	W=np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			temp = 0
			for mu in range(p):
				x_mu = patterns[mu]
				temp = temp + x_mu[i]*x_mu[j]
			W[i,j] = temp
			if scaling == True:
				W[i,j] = W[i,j]/n
	return W
#Bug in update
# def update(x,w):
# 	# X = np.zeros(x.shape)
# 	for mu in range(x.shape[0]):
# 		for i in range(x.shape[1]):
# 			temp=0
# 			for j in range(x.shape[1]):
# 				temp += w[i,j]*x[mu,j]
# 			X[mu] = sign(temp)
# 	print(X)
# 	return X
# 	# for mu in range(x.shape[0]):
# 	# 	for i in range(x.shape[1]):
# 	# 		X[mu]=0
# 	# 		for j in range(x.shape[1]):
# 	# 			X[mu] += w[i,j]*x[mu,j]
# 	# 	 	X[mu] = sign(X[mu])
# 	# return X
def update(x,w):
	X = np.dot(x,w)
	for i in range(len(X)):
		X[i]=sign(X[i])
	return X


def sign(x):
	output = np.where(x>=0,1,-1)
	return output

# def bias(x):
# 	bias = np.ones((x.shape[0],1))
# 	x = np.append(x, bias, axis=1)
# 	return x

patterns = np.array([[-1,-1,1,-1,1,-1,-1,1],[-1,-1,-1,-1,-1,1,-1,-1],[-1,1,1,-1,-1,1,-1,1]], dtype = float)
print(patterns.shape)

# W = Weights(patterns,5)
W = weightMatrix(patterns,None)
print(W)

distorted = np.array([[1,-1,1,-1,1,-1,-1,1],[1,1,-1,-1,-1,1,-1,-1],[1,1,1,-1,1,1,-1,1]], dtype = float)

epochs = 2
X = np.copy(distorted)
for e in range(epochs):
	Xnew = update(X,W)
	if (X == patterns).all():
		print("Stop crietria, itratation: ",e)
		break
	X = np.copy(Xnew)

print(Xnew==patterns)
print(Xnew)
