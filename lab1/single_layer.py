import numpy as np 


#implementation of Delta rule
def Delta_rule(X, T, W):
    """ epoch:
        eta:
        delta_W:
    """
    epoch = 20
    eta = 0.5
    delta_W = -eta*(W*X-T)*X.transpose
    return delta_W

def bias(x):
	bias = np.ones((1,x.shape[1]))
	x = np.append(x, bias, axis=0)
	return x

def firstW(x,t):
	n = x.shape[0]
	m = t.shape[0]
	w = np.random.randn(0,(m,n))
	return w

etha = 0.001
n = 100
mA = [1.0, 0.5]
mB = [-1.0, 0.0]
sigma = 0.5

classA = np.zeros((2,n))
classB = np.zeros((2,n))

classA[0][:] = np.random.randn(1,n) * sigma + mA[0]
classA[1][:] = np.random.randn(1,n) * sigma + mA[1]
classB[0][:] = np.random.randn(1,n) * sigma + mB[0]
classB[1][:] = np.random.randn(1,n) * sigma + mB[1]

X = np.concatenate((classA,classB), axis=1)

t = np.ones((n*2))
t[n:] += -2

X = bias(X)
#perceptron
W = firstW(X,t)

epoch = 20

plt.scatter(classA[0][:],classA[1][:])
plt.scatter(classB[0][:], classB[1][:])
plt.show()
