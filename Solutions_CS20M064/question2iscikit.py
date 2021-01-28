import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
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

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
W_ML=np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),Y)

print(np.shape(W_ML))

#print(linear_regressor.coef_)


#print(np.shape(linear_regressor.coef_))