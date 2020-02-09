import numpy as np

def Weights(x, units):
	W = (x.T).dot(x)
	#W = W/units
	return W
#Bug in update
def update(w,x):
	X = np.zeros_like(x)
	for mu in range(x.shape[0]):
		for i in range(x.shape[1]):
			for j in range(x.shape[1]):
				X[mu] += w[i,j]*x[mu,j]
		X[mu] = signX[mu]
	return X

def sign(x):
	output = np.where(x>=0,1,-1)
	return output

def bias(x):
	bias = np.ones((x.shape[0],1))
	x = np.append(x, bias, axis=1)
	return x

patterns = np.array([[-1,-1,1,-1,1,-1,-1,1],[-1,-1,-1,-1,-1,1,-1,-1],[-1,1,1,-1,-1,1,-1,1]], dtype = np.int8)
print(patterns.shape)

W = Weights(patterns,5)
print(W)

distorted = np.array([[1,-1,1,-1,1,-1,-1,1],[1,1,-1,-1,-1,1,-1,-1],[1,1,1,-1,1,1,-1,1]], dtype = np.int8)

epochs = 100
for e in range(epochs):
	X = update(W, distorted)
	if (X == patterns).all():
		print("Stop crietria, itratation: ",e)
		break
print(X==patterns)
print(X)
