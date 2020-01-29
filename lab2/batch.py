import numpy as np
import math 

def __init__(self,N,node,X,interval):
    N = 5    #dimension of pattern
    node = 3 #number of nodes
    X = 
    interval = [0,2*math.pi]
    # Two functions: sin(2x); square(2x)
    pass

def sin(x):
    return math.sin(2*x)

def square(x):
    if math.sin(2*x)>=0
        return 1
    else 
        return -1
        
def dataGenerate():
    train = np.random.uniform(0,2*math.pi,0.1)
    test = np.random.uniform(0.05,2*math.pi,0.1)
    return train,test

def compute_WeightMatrix():
    # initialize weight matrix
    W0 = []

