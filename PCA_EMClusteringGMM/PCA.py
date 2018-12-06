# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import eig

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
C= M - train
V=np.cov(C.T)

eigenValues, eigenVectors =eig(V)
k=2
idx = eigenValues.argsort()[-k:][::-1]
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]

P = eigenVectors.T.dot(C.T)
PrincipalComponents=P.T



#from sklearn.decomposition import PCA

#pca = PCA(n_components=2)

#principalComponents = pca.fit_transform(train)



plt.scatter(PrincipalComponents[:,0], PrincipalComponents[:,1], color=['red'], marker= '*', s=7)