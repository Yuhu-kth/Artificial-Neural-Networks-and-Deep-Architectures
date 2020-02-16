import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

def sign(x):
	output = np.where(x>=0,1,-1)
	return output

class hopfield():
    def __init__(self, size, npatterns, theta, trainX):
        self.W = np.zeros([npatterns,npatterns])
        self.size = size
        self.npatterns = npatterns
        self.theta = theta
        self.X = trainX
    
    def calc_weights(self):
        rho = 1/(self.size*self.npatterns)*np.sum(self.X)
        for x in self.X:
            self.W += np.outer(x-rho,x-rho)
        for i in range(self.npatterns):
            self.W[i,i] = 0
    
    def update_x(self, x, W):
        print(len(x[0]))
        for i in range(len(x[0])):
            x[0,i] = 0.5 + 0.5*sign(np.dot(x,W[i,:])-self.theta)
        return x

    def update_batch(self,x):
        new = np.dot(x,self.W)-self.theta
        new[new>=0] = 1
        new[new<0] = 0
        return new

    def train(self):       
        self.calc_weights()            


def gen_random_data(dim, length, sparsity):
    train_X = np.random.uniform(0,1,(dim,length))
    train_X[train_X > sparsity] = 1
    train_X[train_X <= sparsity] = 0
    return train_X

def add_noise(image, percentage_flip):
    result = np.copy(image)
    n_flip = int(len(image)*percentage_flip/100)
    units = np.random.choice(len(image),n_flip,replace=False)
    for i in units:
        result[i] = np.abs(result[i]-1)
    return result

def recall_accuracy(input_image,recall_image):
    correct = 0
    counter = 0
    for i in range(len(input_image)):
        if input_image[i] == 1:
            counter += 1
            if input_image[i] == recall_image[i]:
                    correct += 1
    return correct*100/counter

theta_list = [0.0,0.3,0.6,0.9]
for theta in theta_list:
    dim = range(1,101)
    accVals = []
    steps = []
    for d in dim:
        avgAcc = 0
        for i in range(2):
            train_X = gen_random_data(d,1024,0.9)
            testData = train_X[0]
            hopfield_net = hopfield(train_X.shape[0],train_X.shape[1],theta,train_X)
            hopfield_net.train()
            noisyData = add_noise(testData,30)
            testNew = hopfield_net.update_batch(noisyData)
            avgAcc += recall_accuracy(testData,testNew)/2
        accVals.append(recall_accuracy(testData,testNew))
        steps.append(d)

    plt.plot(steps,accVals)
    plt.xlabel('patterns stored')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for sparse (rho=0.01) patterns')
    plt.legend(["0.0","0.3","0.6",'0.9'])
    plt.savefig('result.png')
