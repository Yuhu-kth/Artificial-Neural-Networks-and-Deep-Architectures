import numpy as np 
import matplotlib.pyplot as plt 

def square(x):
	f = []
	y = np.sin(x)
	print(y.shape)
	for ys in y:
		if ys >= 0:
			f.append(1)
		else:
			f.append(-1)
	return f


def sine(x):
	y = np.sin(x)
	return y

def RBF(x, mean, sigma):
	kernel = np.exp((-(x-mean)**2)/(2*(sigma**2)))
	return kernel

def P(x,hidden, sigma, mean):
	Phi = np.zeros((x.shape[0], mean.shape[0]))
	for i, xs in enumerate(x):
		for j, mu in enumerate(mean):
			Phi[i][j] = RBF(xs, mu, sigma)
	return Phi

def leastSquares(Phi, y):
	W = np.linalg.pinv((Phi.T).dot(Phi)).dot(Phi.T).dot(y)
	return W


xTrain = np.arange(0,2*np.pi,0.1).reshape(-1,1)
xTest = np.arange(0.05,2*np.pi, 0.1).reshape(-1,1)

# yTrain = np.sin(2*xTrain)
# yTest = np.sin(2*xTest)
yTrain = square(2*xTrain)
yTest = square(2*xTest)

hidden = np.arange(1,15,1)
sigma = 1

e = []
#Training
for h in hidden:

	arg = np.random.choice(len(xTrain),h)
	mean = xTrain[arg]

	Phi = P(xTrain, h, sigma, mean)
	W = leastSquares(Phi,yTrain)
	#Prediction
	Phi2 = P(xTest, h, sigma, mean)
	y = Phi2.dot(W)
	#-----------remove this for sin(2x) function
	for i in range(len(y)):
		if(y[i]>=0):
			y[i]=1
		else:
			y[i]=-1
	#-------------------------------------------
	error = np.sum(np.abs(y - yTest))/len(y)
	e.append(error)
	if h%2 == 0:
		plt.plot(xTest, y, label = '%i units'%h)

plt.plot(xTest, yTrain,'b',label = 'Real')		
plt.legend()
plt.title("RBF")
plt.show()

plt.plot(np.arange(len(e))+1,e,label='Error')
plt.plot(np.arange(len(e))+1, 0.1*np.ones(len(e)),'--', label='0.1 threshold')
plt.plot(np.arange(len(e))+1, 0.01*np.ones(len(e)),'--', label='0.01 threshold')
plt.plot(np.arange(len(e))+1, 0.001*np.ones(len(e)),'--', label='0.001 threshold')
plt.xlabel('Hidden Nodes')
plt.ylabel('Absolute Error')
plt.legend()
plt.show()
