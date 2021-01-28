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


def getProbablity(mean , x):
    return pow(mean,x)*pow((1-mean), (1-x))


def getLogLikelihood(n, X, mean, pi,lamb):
    logLikelihood=0
    for i in range(n):
        sum=0
        for j in range(k):
            s=pi[j]*(mean[j]**X[i])*(1-mean[j])**(1-X[i])
            if(s!=0):
                s=math.log(s)
            else:
                s=0
            sum+=lamb[i][j]*s
        logLikelihood+=sum
    return logLikelihood

np.set_printoptions(suppress=True)
X = np.array(genfromtxt('A2Q1Data.csv', delimiter=',')).T

n = np.size(X)
k = 4


logLikelihoodList=[]


iteration=0

while(iteration<1):
    iteration+=1
    mean = [0.7,0.3,0,0]

    pi=[0.5, 0.5,0,0]

    lamb=np.zeros([n,k])

    logLikelihoodList.append(getLogLikelihood(n,X,mean,pi,lamb))

    convergence=False
    index=-1
    while(index<100):
        index+=1
        for i in range(n):
            sum=0
            for j in range(k):
                sum+=pi[j]*getProbablity(mean[j],X[i])
            
            for j in range(k):
                if(sum!=0):
                    lamb[i][j]=(pi[j]*getProbablity(mean[j],X[i]))/sum
                if(sum==0):
                    lamb[i][j]=0

        for j in range(k):
            
            sum=0
            size=0
            for i in range(n):
                size+=lamb[i][j]
                sum+=lamb[i][j]*X[i]

            if(size!=0):
                mean[j]=sum/size
            else:
                mean[j]=0
            pi[j]=size/n
        
    
        logLikelihood=getLogLikelihood(n,X,mean,pi,lamb)
       # print(logLikelihood)
        #if(index==0 or logLikelihoodList[index-1]!=logLikelihood):
        logLikelihoodList.append(logLikelihood)


print(pi)

print(mean)

#print(logLikelihoodList)
