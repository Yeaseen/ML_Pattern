#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 00:44:32 2018

@author: yeaseen
"""

import numpy as np


def split(s, delim=[" ", '\n']):
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

def loadfile(filename,checkTrain):
    file = open(filename, "r")
    first = checkTrain
    rows = list()
    for line in file:
        if(first) == True:
            dims = split(line)
            first = False
        else:
            vals = split(line, [' ' ,'\t', '\n'])
            #print(vals)
            rows.append(vals)

    if(checkTrain):
        return dims, rows
    else:
        return rows



dims, rows = loadfile('Train.txt',True)

test = loadfile('Test.txt',False)



dims=np.array(dims)
dims = dims.astype(np.float)
rows=np.array(rows)
mat = rows.astype(np.float)

test=np.array(test)
test= test.astype(np.float)




att=int(dims[0])
#print(att)
clss=int(dims[1])
#print(clss)


wactive =np.random.random_sample(((att+1),))

#wactive=[0,0,1]
#print(w)


matrix=np.array(mat)

Y=matrix[:,-1].copy()
Y=np.array(Y)
Y=Y.astype(np.int)
Y=np.array(Y).tolist()
#print(Y)


matrix[:,-1]=np.ones((matrix.shape[0]))

targetOutput=test[:,-1].copy()
targetOutput=np.array(targetOutput)
targetOutput=targetOutput.astype(np.int)
targetOutput=np.array(targetOutput).tolist()


print(len(targetOutput))


test[:,-1] = np.ones((test.shape[0]))



def checkWithNewWeight(data,weight):
    rowCount=0
    accurate=0
    for row in data:
        prod=np.dot(row,weight)
        if(prod < 0):
            classed=1
        else:
            classed=2
        if(classed == Y[rowCount]):
            accurate+=1
        rowCount+=1
    return accurate



counter=0
maxCounter=matrix.shape[0]

discriminant= True
bestH=0
pocketStart=0
pocketMax=1000
while(discriminant):
    
    countRow=0
    for i in matrix:
        product=np.dot(i,wactive)
        counter+=1
        if(product < 0):
            classed=1
        else:
            classed=2
        if(classed != Y[countRow]):
            pocketStart+=1
            counter =0 
            if(classed == 1):
                #print('sub')
                wactive=np.add(wactive,i)
            else:
                #print('add')
                wactive=np.subtract(wactive,i)
                getH=checkWithNewWeight(matrix,wactive)       
                if(getH > bestH):
                    running_weights=wactive
                    bestH=getH
        countRow+=1
        if((pocketStart==pocketMax) or (counter ==maxCounter) ):
            discriminant=False
    if((pocketStart==pocketMax) or (counter == maxCounter) ):
        discriminant=False
                

print(wactive)


predictedOutput=[]

for eachrow in test:
    val=0
    got=np.dot(eachrow,wactive)
    if(got< 0):
        predictedOutput.append(1)
    else:
        predictedOutput.append(2)

print(predictedOutput)



from sklearn.metrics import classification_report
target_names = []
for i in range(clss):
    target_names.append('class'+str(i))
    


print(classification_report(targetOutput, predictedOutput, target_names=target_names))

from sklearn.metrics import accuracy_score
print("Accuracy is:   "+str(accuracy_score(targetOutput, predictedOutput)))





