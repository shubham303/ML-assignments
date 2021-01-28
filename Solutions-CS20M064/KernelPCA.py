import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math


def getKernelMatrix(X, sigma,n):
    K = np.zeros((n,n))
    sigma=sigma**2
    for i in range(n):
        for j in range(n):
            A=np.subtract(X[:,i],X[:,j])
            A=(np.matmul(A.T,A)*-1)/2*sigma
            K[i][j] = math.exp(A)
        
    return K


def projectData(K,n):
    eig_val, eig_vec = LA.eigh(K)

    eig_vec = eig_vec[:, n-2:n]
    eig_vec = np.matmul(X, eig_vec)

    transformed_data = (np.matmul(eig_vec, np.matmul(eig_vec.T, X)))
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.scatter(transformed_data[0], transformed_data[1], marker='x')
    plt.show()
    
    
X = np.array(genfromtxt(
    'Dataset3.csv', delimiter=','))

X = X.T  # change order to d*n
d = len(X)
n = len(X[0])

for l in (2,3):
    K = np.matmul(X.T, X)
    for i in range(n):
        for j in range(n):
            K[i][j] = pow(K[i][j]+1,l)

    transformed_data=projectData(K,n)
    
   
for sigma in np.arange(0.1,1.1,0.1):
    K=getKernelMatrix(X,sigma,n)
    projectData(K,n)

