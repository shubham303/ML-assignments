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


X = np.matrix(genfromtxt('A2Q2Data_train.csv', delimiter=','))
n=np.size(X)
d=np.size(X[0])-1
Y=X[:,d:]

X=np.delete(X,d,1)
print(np.shape(X))

W=np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),Y)

print(W)
