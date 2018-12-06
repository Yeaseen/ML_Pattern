#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:51:03 2018

@author: yeaseen
"""
import numpy as np
import pandas as pd
import math

class dcsnTreeNodeClass:
    def __init__(self, name):
        self.name = name
        self.lst = {}


def entropyCalc(x,y):
    p1=x/(x+y)
    p2=y/(x+y)
    if(p1==0 or p2==0):
        res=0
    else:
        res=-p1*math.log2(p1)-p2*math.log2(p2)
    return res 

def Importance(attributes,dataframe):
    nodeName=''
    entropy=100
    dataframelength=len(dataframe)
    #print(dataframelength)
    for att in attributes:
        uniqueValues= dataframe[att].unique() 
        #print(uniqueValues)
        res=0
        for value in uniqueValues:
            attvalueFrames=dataframe.loc[dataframe[att] == value]
            subframelength=len(attvalueFrames)
            valueEntropy=subEntropy(attvalueFrames)
            #print(subframelength)
            res+=(subframelength / dataframelength)*valueEntropy
            #print(valueEntropy)
        #print(res,att)
        if(res < entropy):
            #print('once upon a time in mumbai')
            #print(res)
            entropy = res
            nodeName= att       
     
    return nodeName

def pluralityValue(dataframe):
    zeroInOutput=dataframe.iloc[:,-1][dataframe.iloc[:,-1] == -1].count()
    oneInOutput= dataframe.iloc[:,-1][dataframe.iloc[:,-1] == 1].count()
    if(zeroInOutput>= oneInOutput):
        return -1
    else:
        return 1

def classificationFactor(dataframe):
    if(len(dataframe.iloc[:,-1].unique()) == 1):
        #print(dataframe.iloc[:,-1].unique()[0])
        return True
    else:
        return False

def specificClss(dataframe):
    return dataframe.iloc[:,-1].unique()[0]


def dcsnTreeRoot(sampleFrames, attributelist, parentFrames, currDepth, maxDepth):
    if(len(sampleFrames) == 0):
        return pluralityValue(parentFrames)
    elif(classificationFactor(sampleFrames) == True):
        return specificClss(sampleFrames)
    elif(len(attributelist) == 0):
        return pluralityValue(sampleFrames)
    elif(currDepth == maxDepth):
        return pluralityValue(sampleFrames)
    else:
        newnode=Importance(attributelist,sampleFrames)
        print('newnodes =========='+newnode)
        treeRoot= dcsnTreeNodeClass(newnode)
        uniqueValuesRoot=sampleFrames[newnode].unique().tolist()
        uniqueValuesTestRoot=dfTest[newnode].unique().tolist()
        
        uniqeVals= list(set(uniqueValuesRoot+uniqueValuesTestRoot))
        currD=currDepth+1
        if newnode in attributelist : attributelist.remove(newnode)
        #print(attributelist)
        for eachValue in uniqeVals:
            atListCopy=attributelist.copy()
            cuttingFrame=sampleFrames.loc[sampleFrames[newnode] == eachValue]
            subRoot=dcsnTreeRoot(cuttingFrame,atListCopy,sampleFrames,currD,100)
            treeRoot.lst[eachValue] =subRoot
        return treeRoot


def subEntropy(dataframe):
    zeroInOutput=dataframe.iloc[:,-1][dataframe.iloc[:,-1] == -1].count()
    oneInOutput= dataframe.iloc[:,-1][dataframe.iloc[:,-1] == 1].count()
    res=entropyCalc(zeroInOutput,oneInOutput)
    return res


def binningPoint(dataframe):
    #print(dataframe)
    countZero=0
    countOne=0
    entropy=1
    collectorZero=dataframe.iloc[:,-1][dataframe.iloc[:,-1] == -1].count()
    #print(collectorZero)
    collectorOne= dataframe.iloc[:,-1][dataframe.iloc[:,-1] == 1].count()
    #print(collectorOne)
    feature=dataframe.columns[-1]
    total=len(dataframe)
    index=0
    current=0
    for row in dataframe.itertuples(index=True, name='Pandas'):
       # print(getattr(row, feature))
        #print(countZero,countOne,collectorZero,collectorOne)
        if(getattr(row, feature)==-1):
            countZero+=1
            collectorZero-=1
        else:
            countOne+=1
            collectorOne-=1
        #print(countZero,countOne,collectorZero,collectorOne)
        if(not(collectorZero==0 and collectorOne==0)):
            current+=1
            res1= ((countZero+countOne) / total) * entropyCalc(countZero,countOne)
            #print(res1)
            res2= ((collectorZero+collectorOne) / total) * entropyCalc(collectorZero,collectorOne)
            #print(res2)
            #print(res1+res2)
            if(entropy>(res1+res2)):
                index=current
                #print(res1+res2)
                entropy=res1+res2
    #print(dataframe.columns[0])           
    #print(dataframe.iloc[index][dataframe.columns[0]])
    p= dataframe.iloc[index][dataframe.columns[0]]
    q= dataframe.iloc[index+1][dataframe.columns[0]]
    #print(p,q)
    r=(p+q)/2
    return r




df = pd.read_csv('train.csv')

outputList=df[df.columns.values[-1]].unique()
#print(outputList)
df[df.columns.values[-1]]= [-1 if x == outputList[0] else 1 for x in df[df.columns.values[-1]]]

dfTest = pd.read_csv('test.csv')

dfTest[dfTest.columns.values[-1]]= [-1 if x == outputList[0] else 1 for x in dfTest[dfTest.columns.values[-1]]]

testOutput=dfTest.iloc[:,-1].values.tolist().copy()
att=df.drop(df.columns[-1],axis=1).columns.values.tolist().copy()



#print(df)

#print(df.columns.values[-1])

#print(dfTest)






for col_name in df.columns:
    #print(df[col_name].dtypes)
    if df[col_name].dtypes == 'object':
        df[[col_name]] = df[[col_name]].fillna(df[col_name].mode().iloc[0])   
    elif (col_name != df.columns.values[-1]):
        df[[col_name]]=df[[col_name]].fillna(df[col_name].mean())
        new = df.filter([col_name,df.columns[-1]], axis=1)
        new = new.sort_values(col_name)
        #print(new)
        binPoint = binningPoint(new)
        #print(binPoint)
        df[col_name]= [ 0 if x <= binPoint else 1 for x in df[col_name]]


#res = list(set(df['temperature'].unique().tolist()+dfTest['temperature'].unique().tolist()))


#print(res)

#print(df)








#print(df)
print(att)

root = dcsnTreeRoot(df,att.copy(),df,0,1000)

print(root.name)

#print(df)
#att=list(df.drop(df.columns[-1],axis=1).columns.values.copy())


print(att)

root2=dcsnTreeRoot(df,att.copy(),df,0,1000)

print(root2.name)

'''

def classPrint(dataframe,rootOriginal):
    predictOut=[]
    for row in dataframe.itertuples(index=True, name='Pandas'):
        r = rootOriginal
        #print(r.name)
        #print('before while')
        if(np.issubdtype(type(r), int)):
            #print(r)
            predictOut.append(r)
            #break
        else:
            while(1):
                featName=r.name
                #print(featName)
                featValue = getattr(row, featName)
                #print(featValue)
                rootans = r.lst[featValue]
                if(np.issubdtype(type(rootans), int)):
                    #print(rootans)
                    predictOut.append(rootans)
                    break
                else:
                    r=rootans
    return predictOut


#print(classPrint(dfTest,root))
    




def AdaBoost(funcDcsn, sampleFrames, attList, \
             parentFrames,strtdepth, maxdepth,funcClassify,rootClass, K):
    Y=sampleFrames.iloc[:,-1].values.tolist().copy()
    #print(Y)
    N=len(sampleFrames)
    w= [1/N] * N
    h= []
    z= []
    #print(w)
    
    for x in range(0,K):
        data=sampleFrames.sample(n=N,weights=w,replace=True).copy()
        error=0.0001
        root=dcsnTreeRoot(data,attList,data,0,maxdepth)
        predictAns=classPrint(data,root)
        for i in range(0,N):
            if(predictAns[i] != Y[i]):
                error=error + w[i]
        if(error > 0.5):
            #print(x)
            continue
        for j in range(0,N):
            if(predictAns[j] == Y[j]):
                w[j] = w[j] *(error/(1-error))   
        maxVal=sum(w)
        w = [float(i)/maxVal for i in w]
        h.append(root)
        wT= math.log2(((1-error)/error))
        z.append(wT)    
    return h,z

learner, weights=AdaBoost(dcsnTreeRoot,df,att,df,0,1,classPrint,dcsnTreeNodeClass,15)



print(len(learner))

for i in range(len(learner)):
    x=learner[i]
    print(type(x))
    if(np.issubdtype(type(x), int)):
        print(x)
    else:
        print(x.name)


#print(weights)



def learnersAggregation(learner,weights,dfTestData):
    predictOut=[]
    numLearner=len(learner)
    
    for row in dfTestData.itertuples(index=True, name='Pandas'):
        aggAns=0
        for i in range(0,numLearner):
            r=learner[i]
            z=weights[i]
            if(np.issubdtype(type(r), int)):
                aggAns+=r*z
                #print('hello')
            else:
                while(1):
                    featName=r.name
                    #print(featName)
                    featValue = getattr(row, featName)
                    #print(featValue)
                    rootans = r.lst[featValue]
                    if(np.issubdtype(type(rootans), int)):
                        #print(rootans)
                        #predictOut.append(rootans)
                        aggAns+=rootans*z
                        #print('hello')
                        break
                    else:
                        r=rootans
        print(aggAns)
        if(aggAns<0):
            predictOut.append(-1)
        else:
            predictOut.append(1)
    return predictOut        


predictionArr=learnersAggregation(learner,weights,dfTest)


print(predictionArr)

print(testOutput)

from sklearn.metrics import classification_report
target_names = []

for i in range(2):
    target_names.append('class'+str(i))


print(classification_report(testOutput, predictionArr, target_names=target_names))

from sklearn.metrics import accuracy_score
print("Accuracy is      _____:   "+str(accuracy_score(testOutput, predictionArr)))

'''
