import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random

w = np.random.rand(100,84)
w.shape
animals = np.genfromtxt('data/animals.dat',delimiter =',')
len(animals)

f = open('data/animalnames.txt', "r")
animal_names = f.readlines()
f.close()

w = np.random.uniform(0,1,(100,84))
x = animals.reshape((32,84))
x.shape[1]

print(animal_names)

def core_SOM(x, epochs,w):
  
  #Create random matrix  
  n_samples = x.shape[0]
  n_features = x.shape[1]
  n_nodes = w.shape[0]
  distance = np.zeros(n_nodes)
  
  for it in range (epochs):
    N_neighbours = int(50*(1-it/epochs))
    for i in range(n_samples):
      #Compute the minimum node
      for j in range(n_nodes):
        d = np.linalg.norm(x[i,:]-w[j,:])
        distance[j] = d
      winner = np.argmin(distance)
      #Compute the new boundary
      
      min_boundary = max(0,winner - N_neighbours)
      max_boundary = min(100, winner + N_neighbours)
      #Update weights
      
      for j in range(min_boundary, max_boundary):
        w[j] = w[j] + 0.2*(x[i]-w[j])
      
  return w

def SOM_test (x,w):
  
  #Function to test the SOM algorithm
  
  n = x.shape[0]
  indices = []
  n_samples = x.shape[0]
  n_features = x.shape[1]
  n_nodes = w.shape[0]
  distance = np.zeros(n_nodes)
  
  for i in range(n_samples):
      #Compute the minimum node
    for j in range(n_nodes):
        d = np.linalg.norm(x[i,:]-w[j,:])
        distance[j] = d
    winner = np.argmin(distance)
    indices.append(winner)
    
  return np.array(indices)


w = core_SOM(x,20,w)
print(w)

res = SOM_test (x,w)
print(res)


Z = [x for _,x in sorted(zip(res,animal_names))]
print(np.transpose(Z))
