# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 18:10:04 2018

@author: Asus
"""
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
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



def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

def sigmoid_der(x):
    return 1*sigmoid(x)*(1-sigmoid(x))

def derivative(x):
    return x * (1 - x)



train=loadfile('trainNN.txt')

test=loadfile('testNN.txt')

train=np.array(train)
train= train.astype(np.float)

train_output_col=train[:,-1].copy()
train_output_col=np.array(train_output_col)
train_output_col=train_output_col.astype(np.int)
train_output_col=np.array(train_output_col).tolist()

train_Y=train[:,-1].copy()
train_Y=np.array(train_Y)
train_Y = np.eye(np.max(train_Y).astype(int))[train_Y.astype(int)-1]

min_max_scaler = preprocessing.StandardScaler()
train = min_max_scaler.fit_transform(train)
train[:,-1]=np.ones((train.shape[0]))




test=np.array(test)
test= test.astype(np.float)

test_output_col=test[:,-1].copy()
test_output_col=np.array(test_output_col)
test_output_col=test_output_col.astype(np.int)
test_output_col=np.array(test_output_col).tolist()


test_Y=test[:,-1].copy()
test_Y=np.array(test_Y)
test_Y= np.eye(np.max(test_Y).astype(int))[test_Y.astype(int)-1]
test = min_max_scaler.fit_transform(test)
test[:,-1]=np.ones((test.shape[0]))




layerNeurons=[train.shape[1], 3, 4, 5, train_Y.shape[1]]

weight = []
for i in range(len(layerNeurons)-1):
    w = np.random.uniform(-1,1,(layerNeurons[i],layerNeurons[i+1]))
    weight.append(w)



epochRange = 1000
learningRate= 0.01


min_err = np.inf
best_w = []

for epoch in range(epochRange):
    
    for i in range(train.shape[0]):
        v=[]
        y=[]
        inputNeuron = [train[i]]
        v.append(inputNeuron)
        y.append(inputNeuron)
        
        for r in range(len(layerNeurons) -1):
            w=weight[r]
            matout=np.matmul(inputNeuron,w)
            v.append(matout)
            matout=sigmoid(matout)
            y.append(matout)
            inputNeuron=matout
        
        
        lastY=len(y)-1
        errs = 0.5 * (y[lastY] - train_Y[i])*(y[lastY] - train_Y[i])
        if errs.sum() < min_err:
            min_err = errs.sum()
            best_w = weight
        
        delta =[]
        d=(y[lastY] - train_Y[i])* derivative(y[lastY])
        delta.append(d)
        #print(delta[0])
        
        for r in range(len(layerNeurons)-2,0,-1):
            d= (np.matmul(weight[r],delta[len(layerNeurons)-r-2].T)).T
            d= d*derivative(y[r])
            delta.append(d)
        
        
        delta.reverse()
        
        delw=[]
        for i in range(len(layerNeurons)-1):
            w = np.random.uniform(0,0,(layerNeurons[i],layerNeurons[i+1]))
            delw.append(w)
        
        for r in range(len(delw)-1,0,-1):
            delw[r]=np.matmul(np.array(y[r]).T,delta[r])
            
        
        for i in range(len(weight)):
            weight[i] -= learningRate * delw[i]



weight = best_w
output = []
for i in range(train.shape[0]):   
    inputNeuron = [train[i]]
    for r in range(len(layerNeurons)-1):
        w=weight[r]
        matout=np.matmul(inputNeuron,w)
        matout=sigmoid(matout)
        inputNeuron=matout
    
    output.append(np.argmax(inputNeuron)+1)
    
#print(output)


print("Accuracy on Train Data set:   "+str(accuracy_score(train_output_col, output)))


output = []
for i in range(test.shape[0]):   
    inputNeuron = [test[i]]
    for r in range(len(layerNeurons)-1):
        w=weight[r]
        matout=np.matmul(inputNeuron,w)
        matout=sigmoid(matout)
        inputNeuron=matout
    
    output.append(np.argmax(inputNeuron)+1)

print("Accuracy is on Test Data Set:   "+str(accuracy_score(test_output_col, output)))










