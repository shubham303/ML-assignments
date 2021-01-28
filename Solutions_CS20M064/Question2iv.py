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


# fuction to calulate average error
def calculateAverageError(X, Y, W, lamb=0):
    n = len(X)
    A = np.matmul(X, W)
    B = np.subtract(A, Y)
    c = np.matmul(B.T, B)
    error = c/n
    return error[0]



# function to find gradient with given lambda
def gradient(covariance_matrix, B, W, lamb=0):
    n = len(X)
    C = np.matmul(covariance_matrix, W)
    D = np.subtract(C, B)
    E = lamb*W
    gradient = np.add(D, E)
    return gradient

# calculate L2 distance between two vectors
def distance(A, B):
    i = 0
    n = len(A)
    distance = 0
    while(i < n):
        distance += (A[i]-B[i])**2
        i += 1
    D = math.sqrt(distance)
    return D


X = np.array(genfromtxt('A2Q2Data_train.csv', delimiter=','))
n = np.size(X)
d = np.size(X[0])-1
Y = X[:, d:]

X = np.delete(X, d, 1)


# divide data set between training data and validation data . 70 percent data is in training.
validation_set_X = X[7000:]
validation_set_Y = Y[7000:]

X = X[0:7000]
Y = Y[0:7000]

# total number of iterations 
total_iterations = 10000
stepSize = 0.000001

# hyperparameter of normal regression .
W_ML = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)

covariance_matrix = np.matmul(X.T, X)
B = np.matmul(X.T, Y)




error_dist = [0]*100
lamb=0.0
# loop over different lambda values.
for i in range(0,100):
    t = 0
    W = np.zeros((d, 1))
    while(t < total_iterations):
        F = stepSize*gradient(covariance_matrix, B, W, lamb)
        Z = np.subtract(W, F)
        W = Z
        t += 1
        
    error_dist[i] =calculateAverageError(validation_set_X, validation_set_Y, W, lamb)
    if(i==0 or error_dist[i]<error_dist[i-1]):
        W_R=W
        min_lambda=lamb
    lamb+=0.05

print(min_lambda)

test_X = np.array(genfromtxt('A2Q2Data_train.csv', delimiter=','))
n = np.size(test_X)
d = np.size(test_X[0])-1
test_Y = test_X[:, d:]

test_X = np.delete(test_X, d, 1)

print("average error of W_R:{}".format(calculateAverageError(test_X, test_Y,W_R)))
print("average error of W_ML:{}".format(calculateAverageError(test_X, test_Y,W_ML)))

plt.xlabel("lamda")
plt.ylabel("error")
plt.plot(np.arange(0.05,5.05,0.05),error_dist)
plt.show()
