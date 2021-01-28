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

def distance(A,B):
    i=0
    n=len(A)
    distance=0
    while(i<n):
        distance+=(A[i]-B[i])**2
        i+=1
    D=math.sqrt(distance)
    return D

X = np.array(genfromtxt('A2Q2Data_train.csv', delimiter=','))
n=np.size(X)
d=np.size(X[0])-1
Y=X[:,d:]

X=np.delete(X,d,1)

T=5000
t=0
stepSize=0.0001

error_dist=[0]*T

W=np.zeros((d,1))
W_ML=np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),Y)

A=np.matmul(X.T,X)
B=np.matmul(X.T,Y)
while(t<T):
    C=np.matmul(A,W)
    D=stepSize*np.subtract(C,B)
    Z= np.subtract(W,D)
    W=Z
    error_dist[t]=distance(W_ML , W) 
    t+=1


plt.xlabel("iteration")
plt.ylabel("distance of W from W_ML")
plt.plot(range(0,T),error_dist)
plt.show()
