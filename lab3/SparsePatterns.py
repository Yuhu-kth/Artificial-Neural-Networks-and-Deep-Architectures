import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

def sign(x):
	output = np.where(x>=0,1,-1)
	return output

class hopfield():
    def __init__(self, size, npatterns, theta, trainX):
        self.W = np.zeros([npatterns, npatterns]) # bias term
        self.size = size            #number of units
        self.npatterns = npatterns  #number of patterns
        self.theta = theta
        self.X = trainX
    
    def Weights(self):
        rho = 1/(np.dot(self.size,self.npatterns))*sum(self.X)
        W = np.zeros((self.size,self.size))
        for x in self.X:
            self.W += np.outer((x-rho),(x-rho))
        for i in range(self.npatterns):
            self.W[i,i] = 0
        # print("W",self.W)

    
    def update(self,x):
        # print(x.shape)
        temp = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                temp += np.dot(x[j],self.W[i,j])
                x[i] = 0.5+0.5*sign(temp-self.theta)
        return x


def generatePattern(dim, length, sparsity):
        trainX = np.random.uniform(0,1,(dim,length))
        trainX[trainX>sparsity] = 1
        trainX[trainX<sparsity] = 0
        return trainX
        

def Accuracy(inputData, outputData):
    correct = 0
    counter = 0
    for i in range(len(inputData)):
        if inputData[i] == 1:
            counter += 1
            if inputData[i] == outputData[i]:
                counter += 1
                if inputData[i] == outputData[i]:
                    correct += 1
        return correct*100/counter


thetaList = [0.0, 0.3, 0.6, 0.9]
for theta in thetaList:
    dim = range(1,100)
    accuracy = []
    Npatterns = []
    for d in dim :
        avgAcc = 0
        for i in range(2):
            trainX = generatePattern(d, 1024, 0.9)
            hopfield_frame= hopfield(trainX.shape[0],trainX.shape[1],theta,trainX)
            testData = trainX[0]
            hopfield_frame.Weights()
            testNew = hopfield_frame.update(testData)
            avgAcc += Accuracy(testData,testNew)/2
        accuracy.append(avgAcc)
        Npatterns.append(d)

plt.plot(Npatterns,accuracy)
plt.xlabel("Number of Patterns")
plt.ylabel("Accuracy")
plt.legend(['0.0','0.3','0.6','0.9'])
plt.savefig('r.png')

        

