# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import itertools
from numpy.linalg import eig
from scipy.stats import multivariate_normal
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl


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


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], s=7, color=color, marker= '*')

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-10., 10.)
    plt.ylim(-10., 10.)
    #plt.xticks(())
    #plt.yticks(())
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.show()



train=loadfile('onlineDataset.txt')
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
plt.xlim(-10., 10.)
plt.ylim(-10., 10.)
plt.title('Scatter plot DATASET after PCA')
plt.show()




## ALL INITIALISATIONS 

numOfGauss=4  ## it is determined from the above plot

color_iter = itertools.cycle(['red', 'green', 'blue','yellow'])


#initial mean, cov, w
mu = np.random.uniform(min(PrincipalComponents[:,0]),max(PrincipalComponents[:,0]),\
                       size=(numOfGauss,len(PrincipalComponents[0])))

cov = np.zeros((numOfGauss,len(PrincipalComponents[0]),len(PrincipalComponents[0])))
## it must be remembered that any of them shouldnt be a singular matrix, so we fill main diagonal
## line witha a value
for dim in range(len(cov)):
            np.fill_diagonal(cov[dim],5.00)
            
w=[1/numOfGauss]*numOfGauss


N=PrincipalComponents.shape[0]
epsilon=1e-6
old_log_likelihood = 0
probs =np.zeros((N, numOfGauss), np.float)
iterationNo=0
while(True):
    
    #GAUSSIAN MULTIVARIATE FINDING
    norm_columns=[]
    for i in np.arange(numOfGauss):
        ans=np.array(multivariate_normal.pdf(PrincipalComponents, mu[i], cov[i]))
        norm_columns.append(ans)
           
    norm_densities=np.column_stack(norm_columns)
    
    ## LOGLIKELIHOOD calculation
    innerSumVector=np.log(np.array([np.dot(np.array(w).T,norm_densities[i]) for i in np.arange(N)]))
    
    log_likelihood = np.dot(innerSumVector.T, np.ones(N))
    
    ## END CHECK
    if(np.absolute(log_likelihood - old_log_likelihood) < epsilon):
        break
    ## E STEP
    counter=0
    for i in norm_densities:
        mul=i*w
        sumrow=np.sum(mul)
        mul=mul/sumrow
        probs[counter]=mul
        counter+=1
    
    ## M step
    for i in range(numOfGauss):
        probabilty=(probs.T)[i]
        denominator= np.dot(probabilty.T, np.ones(N))
        mu[i] = np.dot(probabilty.T,PrincipalComponents) / denominator
        diff=PrincipalComponents - np.tile(mu[i], (N, 1))
        cov[i]=np.dot(np.multiply(probabilty.reshape(N,1),diff).T,diff) / denominator
        w[i]= denominator / N
    
    old_log_likelihood=log_likelihood
    
    ## MAXEXPECTAION MODEL FINDS FOR EACH ROW
    MaxExpectations=[]
    for i in probs:
        MaxExpectations.append(np.argmax(i))
    MaxExpectations=np.array(MaxExpectations)
    
    ##Plotting at each iteration
    plot_results(PrincipalComponents, MaxExpectations, mu, cov, 0,
             'Gaussian Mixture Model at iteration ' + str(iterationNo))
    iterationNo+=1

print(np.array(w)*N)












