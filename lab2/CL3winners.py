import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def square(x):
	f = []
	y = np.sin(x)
	for ys in y:
		if ys >= 0:
			f.append(1)
		else:
			f.append(-1)
	return f

def RBF(x, mean, sigma):
	kernel = np.exp((-dist(x,mean)**2)/(2*(sigma**2)))
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

def delta(x,y,learning,epoch, mean,hidden):
	W = np.random.randn(hidden, xTrain.shape[1])
	for i in range(epoch):
		for j,xs in enumerate(x):
			Phi = P(xs,hidden,sigma,mean)
			W += learning * (y[i] - (Phi.T).dot(W)).dot(Phi)
	return W

def winner(x, weight):
	win = weight[0]
	ind = 0
	distance = []
	for ws in weight:
		distance.append(x-ws)
	ind = np.argmin(distance)
	win = weight[ind]
	return win, ind
	
def winner2(x,weight, winners):
	w = []
	i = []
	h = []
	ind = 0
	distance = []
	win = weight[0]
	for ws in weight:
		distance.append(dist(x,ws))
	for ww in range(winners):
		ind = np.argmin(distance)
		win = weight[ind]
		i.append(ind)
		w.append(win)
		h.append(neighborhood(distance[ind],sigma))
		distance[ind] = 10000	

	return w,i,h

def neighborhood(distance, sigma):
	h = np.exp(-(distance**2)/(2*sigma**2))
	return h

def dist(x,y):
	x1,y1 = x
	x2,y2 = y
	return np.sqrt((x2-x1)**2 + (y2-y1)**2)

data = make_blobs(n_samples = 200, n_features = 2, centers = 1, random_state = 2)

X = data[0]
Y = data[1]
print(X.shape)

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

np.random.seed(1)
hidden = np.arange(2,11,1)
sigma = 1
epoch = 10000
eta = 0.001
e = []
Win = 3
#Training
for h in hidden:
	print("Hidden units: ",h)
#h = 10
	arg = np.random.choice(len(X),h)
	mean = X[arg]

	for ep in range(epoch):
		sample = np.random.choice(X.shape[0])
		#win, ind = winner(sample,mean)
		win, ind, h = winner2(X[sample], mean, Win)

		et = eta * np.ones_like(win)
		for l in range(len(win)):
			win[l] = win[l] + et[l] * h[l] * (dist(X[sample],win[l]))

		for j,ix in enumerate(ind):
			mean[int(ix)] = win[j]
		#if ep%100 == 0:
	plt.scatter(X[:,0],X[:,1] , c = data[1], cmap = 'viridis')
	plt.scatter(X[arg,0],X[arg,1] , label = 'Initial Units', c = 'red')
	plt.scatter(mean[:,0], mean[:,1], label = 'Final Units', c = 'orange')
	plt.legend()
	plt.title("Competitive Learning, %i Winners"%Win)
	plt.show()

	Phi = P(X, h, sigma, mean)
	W = leastSquares(Phi,Y)
	#Prediction
	"""	Phi2 = P(xTest, h, sigma, mean)
	y = Phi2.dot(W)"""

	"""	error = np.sum(np.abs(y - yTest))/len(y)
	e.append(error)"""
	


"""plt.plot(hidden,e,label='Error')
plt.plot(hidden, 0.1*np.ones(len(e)),'--', label='0.1 threshold')
plt.plot(hidden, 0.01*np.ones(len(e)),'--', label='0.01 threshold')
plt.plot(hidden, 0.001*np.ones(len(e)),'--', label='0.001 threshold')
plt.xlabel('Hidden Nodes')
plt.ylabel('Absolute Error')
plt.legend()
plt.show()
"""






plt.show()
