#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:56:26 2018

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
#print(test)
#print(dims)
#print(mat)







'''
matrix=[[11.0306  ,  9.0152  ,  8.0199	, 7.62	,	1],
        [11.4008  ,  8.7768  ,  6.7652	, 8.9	,	1],
        [14.6263  ,  1.8092 ,  10.7134	, 6.7	,	2],
        [15.4467   , 1.2589  , 12.6625	, 8.2	,	2],
        [1.3346  ,  5.8191  ,  2.0320	, 7.5	,	3],
        [0.7506 ,   5.6132  ,  0.9061	, 7.7	,	3]
        ]
matrixOutput=[11.0306  ,  9.0152  ,  8.0199 , 7.62	]
   '''     
#matrixOutput=matrixOutput+ [1]

att=int(dims[0])
#print(att)
clss=int(dims[1])
#print(clss)

finalMat=np.empty((0,(att+1)*clss))
w =np.random.random_sample(((att+1)*clss,))
#print(w)

matrix=np.array(mat)

Y=matrix[:,-1].copy()
Y=np.array(Y)
Y=Y.astype(np.int)
Y=np.array(Y).tolist()
#print([Y])
matrix[:,-1]=np.ones((matrix.shape[0]))

targetOutput=test[:,-1].copy()
targetOutput=np.array(targetOutput)
targetOutput=targetOutput.astype(np.int)
targetOutput=np.array(targetOutput).tolist()

test[:,-1] = np.ones((test.shape[0]))

#print(test)
#print(matrix)
count=0
for i in matrix:
    #print(i)
    a=np.zeros(((att+1)*clss))
    #print(Y[count])
    #print(count)
    classVal=int(Y[count])
    #print(classVal)
    a[(classVal-1)*(att+1) : classVal*(att+1)]=i
    #print([a])
    for j in range(clss):
        if( (j+1) != classVal):
            x=a.copy()
            x[j*(att+1) : (j+1)*(att+1)] = -i
            finalMat = np.vstack([finalMat, x])
            #print(x)
    count+=1


#print(finalMat)    
        
counter=0
maxCounter=finalMat.shape[0]
constantTerm= 0.5
discriminant= True
while(discriminant):
    for i in finalMat:
        product=np.dot(i,w)
        counter+=1
        if(product < 0):
            w=w+(i*constantTerm)
            #print(w)
            counter = 0
        #print(product)
        if(counter == maxCounter):
            discriminant = False

#print(w)
            
predictedOutput=[]
for eachrow in test:
    val=0
    classed=0
    for k in range(clss):
        a=np.zeros(((att+1)*clss))
        a[k*(att+1) : (k+1)*(att+1)] = eachrow
        got=np.dot(a,w)
        if(val<got):
            val=got
            classed=k+1
    predictedOutput.append(classed)
    #print(classed)
    

#print(predictedOutput)


from sklearn.metrics import classification_report
target_names = []
for i in range(clss):
    target_names.append('class'+str(i))
    


print(classification_report(targetOutput, predictedOutput, target_names=target_names))

from sklearn.metrics import accuracy_score
print("Accuracy is:   "+str(accuracy_score(targetOutput, predictedOutput)))





'''
print(matrixOutput)

matrixOutput = np.array(matrixOutput)

for k in range(clss):
    a=np.zeros(((att+1)*clss))
    a[k*(att+1) : (k+1)*(att+1)] = matrixOutput
    #print(a)
    print(np.dot(a,w))

     '''   