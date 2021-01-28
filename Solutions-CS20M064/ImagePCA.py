
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

def eucledianDistance(transformed_data,test_vector,category_list):
    distance=[]
    distance.append(0)
    distance.append(0)
    category_one_count=0 
    category_two_count=0
    for i in range(n):
        if(category_list[i]==1):
            category_one_count+=1
            distance[0]+=np.linalg.norm(transformed_data[:,i]-test_vector)
        else:
            category_two_count+=1
            distance[1]+=np.linalg.norm(transformed_data[:,i]-test_vector)
    distance[0]/=category_one_count
    distance[1]/=category_two_count
    return distance



images = glob.glob('images/*.JPG')
X = np.empty((0, 1048576))
category_list=[]      #category data point belongs

for image in images:
    data = np.array(Image.open(image)).flatten()
    X = np.vstack([X, data])
    if("vessel" in image):
        category_list.append(1)
    else:
        category_list.append(2)

X = X.T

d = len(X)
n = len(X[0])

mean = np.mean(X, axis=1)

for i in range(n):
    X[:,i]=np.subtract(X[:,i],mean)


K = np.matmul(X.T, X)
K = K*1/n

eig_val, eig_vec = LA.eigh(K)

eig_vec = np.matmul(X, eig_vec)

for i in range(n):
    if(n*eig_val[i]>0):
        c=math.sqrt(n*eig_val[i])
        eig_vec[:, i] = eig_vec[:, i]/c

for k in (4,8,12,20,30,40):
    print("size of eigen subspace={}".format(k))
    w =eig_vec[:,n-k:n]
    transformed_data = (np.matmul(w, np.matmul(w.T, X)))

    for i in range(n):
        transformed_data[:, i] = np.add(mean, transformed_data[:,i]  )
    
    images = glob.glob('test-images/*.JPG')
    index=0
    for image in images:
        index+=1
        category=0
        if("vessel" in image):
            category = 1
        else:
            category = 2

        test_data = np.array(Image.open(image)).flatten()
        test_data = np.subtract(test_data, mean)
        test_data = (np.matmul(w, np.matmul(w.T, test_data)))
        test_data=np.add(test_data,mean)
        distance=eucledianDistance(transformed_data,test_data,category_list)
        print("category of test data point {} is :{}".format(index,category))
        print("distance of first test data point from category 1={}".format(distance[0]))
        print("distance of first test data point from category 2={}".format(distance[1]))
        print("*********************************")
    print("+++++++++++++++++++++++++++")