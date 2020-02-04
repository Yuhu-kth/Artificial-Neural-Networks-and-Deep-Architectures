import numpy as np 
import math
import matplotlib.pyplot as plt 

def square(x):
	f = []
	y = np.sin(x)
	# print(y.shape)
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

def P(x,hidden,sigma,mean):
	Phi = np.zeros((hidden,1))
	for i, xs in enumerate(x):
		for j, mu in enumerate(mean):
			Phi[i][j] = RBF(xs, mu, sigma)
	return Phi

def leastSquares(Phi, y):
	W = np.linalg.pinv((Phi.T).dot(Phi)).dot(Phi.T).dot(y)
	return W

def Delta_rule(etha,phi_Xk):
    """ 
    """
    delta_W = np.dot(np.dot(etha,math.e),phi_Xk)
    return delta_W

def firstW(n):
    """initialization of weight matrix
    
    :param n: number of the row of inputs matrix
    :type n: int
    :param m: number of the row of output matrix
    :type m: int
    :return: weight matrix
    :rtype: array
    """
    w = np.random.randn(n)
    return w

def sequential(numberHidden):
    #add noise
    xTrain = np.arange(0,2*np.pi,0.1)
    xTest = np.arange(0.05,2*np.pi,0.1)
    noiseTrain = np.random.normal(0,0.1,xTrain.shape[0])
    noiseTest = np.random.normal(0,0.1,xTest.shape[0])
    xTrain = xTrain+noiseTrain
    xTest = xTest+noiseTest
    # yTrain = np.sin(2*xTrain)+noise
    # yTest = np.sin(2*xTest)+noise
    yTrain = square(2*xTrain)
    yTest = square(2*xTest)
    # hidden = np.arange(1,15,1)
    sigma = 1
    epoch = 3
    etha = 0.001
    e = []
    error = []
    Phi = np.zeros((xTrain.shape[0],numberHidden))
    Phi2 = np.zeros((xTest.shape[0],numberHidden))
    W = firstW(numberHidden)
    #Training

    for i in range(epoch):
        for h in range(numberHidden):
            arg = np.random.choice(len(xTrain),h)
            mean = xTrain[arg]
            for j in range(xTrain.shape[0]):
                for k in range(h):
                    Phi[j,k] = P(xTrain, h, sigma, mean)
                W += Delta_rule(etha,Phi).reshape(1,-1)
            #prediction
            for i in range(h):
                Phi2[:,i] = P(xTest, h, sigma, mean)
            y = Phi2*W
                # -----------remove this for sin(2x) function
            for i in range(len(y)):
                if(y[i]>=0):
                    y[i]=1
                else:
                    y[i]=-1
                # -------------------------------------------
            error = np.sum(np.abs(y - yTest))/len(y)
            e.append(error)
            error = []
            print(error)

sequential(5)        







#     if h%2 == 0:
#         plt.plot(xTest, y, label = '%i units'%h)

# plt.plot(xTest, yTrain,'b',label = 'Real')		
# plt.legend()
# plt.title("RBF")
# plt.show()

# plt.plot(np.arange(len(e))+1,e,label='Error')
# plt.plot(np.arange(len(e))+1, 0.1*np.ones(len(e)),'--', label='0.1 threshold')
# plt.plot(np.arange(len(e))+1, 0.01*np.ones(len(e)),'--', label='0.01 threshold')
# plt.plot(np.arange(len(e))+1, 0.001*np.ones(len(e)),'--', label='0.001 threshold')
# plt.xlabel('Hidden Nodes')
# plt.ylabel('Absolute Error')
# plt.legend()
# plt.show()
