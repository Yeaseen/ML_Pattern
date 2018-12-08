# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import eig
from scipy.stats import multivariate_normal
def split(s, delim):
    words = []
    word = []
    for c in s:
        if c not in delim:
            word.append(c)
        else:
            if word:
                words.append(''.join(word))
                word = []
    if word:
        words.append(''.join(word))
    return words

def loadfile(filename):
    file = open(filename, "r")
    rows = list()
    for line in file:
        vals = split(line, [' ' ,'\t', '\n'])
        rows.append(vals)
    return rows



train=loadfile('data.txt')
train=np.array(train)
train= train.astype(np.float)


M = np.mean(train.T, axis=1)
C= M -train 
V=np.cov(C.T)

eigenValues, eigenVectors =eig(V)
top_vectors=2
idx = eigenValues.argsort()[-top_vectors:][::-1]
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]

P = eigenVectors.T.dot(C.T)
PrincipalComponents=P.T

plt.scatter(PrincipalComponents[:,0], PrincipalComponents[:,1], color=['red'], marker= '*', s=7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scatter plot DATASET')
plt.show()

numOfGauss=3
#initial mean, cov, w
mu = np.random.randint(min(PrincipalComponents[:,0]),max(PrincipalComponents[:,0]),\
                       size=(numOfGauss,len(PrincipalComponents[0])))

cov = np.zeros((numOfGauss,len(PrincipalComponents[0]),len(PrincipalComponents[0])))


for dim in range(len(cov)):
            np.fill_diagonal(cov[dim],np.random.random_sample())
            

w=[1/numOfGauss]*numOfGauss
N=PrincipalComponents.shape[0]



plt.scatter(PrincipalComponents[:,0], PrincipalComponents[:,1], color=['black'], marker= '*', s=7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scatter plot DATASET')


n=[]
for i in np.arange(numOfGauss):
    ans=np.array(multivariate_normal.pdf(PrincipalComponents, mu[i], cov[i]))
    plt.scatter(mu[i][0],mu[i][1])
    n.append(ans)
    
norm_densities=np.column_stack(n)

plt.show()

















