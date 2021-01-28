import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

np.set_printoptions(suppress=True)
X = np.array(genfromtxt('Dataset3.csv', delimiter=','))

X = X.T  # change order to d*n
d=len(X)
n=len(X[0])
print(np.mean(X,axis=1))
# PCA without data centering
C=np.matmul(X,X.T)
C=C*1/n
eig_val, eig_vec=LA.eigh(C)
print("PCA results without data centering")
print("eigenvalue: {}".format(eig_val))
print("eigen vectors in columns: {}".format(eig_vec))

### PCA implementation With  data centering
mean = np.mean(X,axis=1)
for i in range(len(X[0])):
    X[:,i]=np.subtract(X[:,i],mean)
    
C=np.matmul(X,X.T)
C=C*1/n
eig_val, eig_vec=LA.eigh(C)
print("*******************************")
print()
print("PCA results With data centering")
print("eigenvalue: {}".format(eig_val))
print("eigen vectors in columns: {}".format(eig_vec))