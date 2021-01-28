
import glob
from PIL import Image
from numpy import asarray
import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib
import math
import random

np.set_printoptions(suppress=True)
X = np.array(genfromtxt('A2Q1Data.csv', delimiter=',')).T

print(np.size(X))
n = np.size(X)
k = 4

z = [0]*n
mean = [0]*k


size_of_clusters = [0]*k

totalError = []

sum = [0]*k

for i in range(0, n):
    a = random.randint(0, k-1)
    z[i] = a

convergence = False

iterations=0
while(not convergence):
    iterations+=1
    size_of_clusters = [0]*k
    sum = [0]*k
    convergence = True

    for i in range(0, n):
        size_of_clusters[z[i]] += 1
        sum[z[i]] += X[i]

    for i in range(k):
        if(sum[i] != 0):
            mean[i] = sum[i]/size_of_clusters[i]
        else:
            mean[i] = 0

    error=0
    for i in range(n):
        error+=(mean[z[i]]-X[i])**2
    totalError.append(error)


    for i in range(n):
        minDist = (mean[0]-X[i])**2
        cluster = 0
        for j in range(1, k):
            dist = (mean[j]-X[i])**2
            if(dist < minDist):
                minDist = dist
                cluster = j

        if( convergence and z[i] != cluster):
            convergence = False
        z[i] = cluster


print(mean)
print((totalError))
print(iterations)


plt.xlabel("iteration")
plt.ylabel("mean squared error")
plt.plot( range(0, len(totalError),1),totalError, marker='x')
plt.show()