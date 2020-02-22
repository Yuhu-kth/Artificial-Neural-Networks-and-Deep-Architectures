import numpy as np 
import matplotlib.pyplot as plt
import math

def dataGenerate():
    XTrain = np.random.uniform(0,2*math.pi,0.1)
    XTest = np.random.uniform(0.05,2*math.pi,0.1)
    return XTrain,XTest

def phi(x,miu,sigma):
    phi = np.exp((-(x-miu)**2)/(2*(sigma**2))
    return phi

def sin(x):
    return np.sin(x)

def square(x):
    if np.sin(2*x) >= 0:
        return 1
    else:
        return -1

def leastSquare():
    phi



