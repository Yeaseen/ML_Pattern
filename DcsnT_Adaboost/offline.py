#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:49:22 2018

@author: yeaseen
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
        #print('newnodes =========='+newnode)
        treeRoot= dcsnTreeNodeClass(newnode)
        uniqueValuesRoot=sampleFrames[newnode].unique().tolist()
        uniqueValuesTestRoot=dfTestN[newnode].unique().tolist()
        
        uniqeVals= list(set(uniqueValuesRoot+uniqueValuesTestRoot))
        currD=currDepth+1
        if newnode in attributelist : attributelist.remove(newnode)
        #print(attributelist)
        #print('befor for loop')
        for eachValue in uniqeVals:
            atListCopy=attributelist.copy()
            cuttingFrame=sampleFrames.loc[sampleFrames[newnode] == eachValue]
            subRoot=dcsnTreeRoot(cuttingFrame,atListCopy,sampleFrames,currD,maxDepth)
            treeRoot.lst[eachValue] =subRoot
        #print('after for loop')
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



def dataCleaning(dataframe):
    for col_name in dataframe.columns:
        if(dataframe[col_name].dtypes == 'object'):
            dataframe[[col_name]] = dataframe[[col_name]].fillna(dataframe[col_name].mode().iloc[0])   
        elif(col_name != dataframe.columns.values[-1]):
            dataframe[[col_name]]=dataframe[[col_name]].fillna(dataframe[col_name].mean())
            new = dataframe.filter([col_name,dataframe.columns[-1]], axis=1)
            new = new.sort_values(col_name)
            binPoint = binningPoint(new)
            dataframe[col_name]= [ 0 if x <= binPoint else 1 for x in dataframe[col_name]]
    return dataframe



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
        attList=list(data.drop(data.columns[-1],axis=1).columns.values)
        roott=dcsnTreeRoot(data,attList.copy(),data,0,maxdepth)
        predictAns=classPrint(data,roott)
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
        h.append(roott)
        #print(type(roott))
        wT= math.log2(((1-error)/error))
        z.append(wT)    
    return h,z


def learnersAggregation(learner,weights,dfTestData):
    predictOutAgg=[]
    numLearner=len(learner)
   # print(numLearner)
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
        #print(aggAns)
        if(aggAns<0):
            predictOutAgg.append(-1)
        else:
            predictOutAgg.append(1)
    return predictOutAgg        


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]== -1:
           TN += 1
        if y_hat[i]== -1 and y_actual[i]!=y_hat[i]:
           FN += 1
    
    return (TP, FP, TN, FN)


def printMeasure(TP,FP,TN,FN):
    print('sensitivity,recall,hitrate: ')
    TPR = TP/(TP+FN)
    print(TPR)
    print('specificity: ')
    TNR = TN/(TN+FP)
    print(TNR)
    print('Precision: ')
    PPV = TP/(TP+FP)
    print(PPV)
    print('false discovery rate: ')
    FDR = FP/(TP+FP)
    print(FDR)
    print('accuracy: ')
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print(ACC)
    print('fi score: ')
    FI_SCORE= 2*((PPV*TPR)/(PPV+TPR))
    print(FI_SCORE)




'''

###preproecesing od ADULT DATA SET
df = pd.read_csv('adult_train.csv',skipinitialspace=True)


df[df.columns.values[-1]]= [-1 if x == '<=50K' else 1 for x in df[df.columns.values[-1]]]

dfTest = pd.read_csv('adult_test.csv',skipinitialspace=True)

dfTest[dfTest.columns.values[-1]]= [-1 if x == '<=50K' else 1 for x in dfTest[dfTest.columns.values[-1]]]


df['native_country'] = ['United-States' if x == 'United-States' else 'Other' for x in df['native_country']]


dfTest['native_country'] = ['United-States' if x == 'United-States' else 'Other' for x in dfTest['native_country']]



dfN=dataCleaning(df)
dfTestN=dataCleaning(dfTest)

###end of ADULT DATASET



'''

###preprocessing of CREDIT CARD FRAUD DETECTION DATA

df = pd.read_csv('creditcard.csv')

isFraud = df[df['Class'] == 1]
#print(len(isFraud))
isNotFraud= df.iloc[100:1400]
isNotFraud= isNotFraud[isNotFraud['Class'] == 0 ]
#print(len(isNotFraud))





frames=[isNotFraud,isFraud]

result= pd.concat(frames)

#print(len(result))

dfFull=dataCleaning(result)
dfFull[dfFull.columns.values[-1]]= [-1 if x == 0 else 1 for x in dfFull[dfFull.columns.values[-1]]]
dfN,dfTestN= train_test_split(dfFull,test_size=0.2,shuffle=True,)


#for col_name in dfN.columns:
   # unique_cat = len(dfN[col_name].unique())
   # print(col_name)
    #print("Feature '{col_name}' has {unique_cat} unique categories\
      #        ".format(col_name=col_name, unique_cat=unique_cat))


### end of preprocessing of CREDIT CARD FRAUD DETECTION DATA










## WORKING SCENARIO STARTED HERE
#print(dfN['Class'].value_counts())
#print(dfTestN['Class'].value_counts())

#print(dfN.info())

#print(dfTestN.info())
#print(df.head(5))
#print(dfTest.head(5))







###starting of table1
Yaw=dfN.iloc[:,-1].values.tolist().copy()
testOutput=dfTestN.iloc[:,-1].values.tolist().copy()


#print(len(Yaw))
#print(dfN.head(5))
#print(dfTestN.head(5))



att=list(dfN.drop(dfN.columns[-1],axis=1).columns.values)
print(len(att))

root=dcsnTreeRoot(dfN,att.copy(),dfN,0,15)

#print(type(root))


#print(len(testOutput))

predictOutputTrain=classPrint(dfN,root)

#print(len(predictOutputTrain))

predictedOutputTest=classPrint(dfTestN,root)

#print(len(predictedOutputTest))


TP,FP,TN,FN=perf_measure(Yaw,predictOutputTrain)
print("===========Over train DATA =========")
printMeasure(TP,FP,TN,FN)



print("===========Over test DATA =========")
TP,FP,TN,FN=perf_measure(testOutput,predictedOutputTest)
printMeasure(TP,FP,TN,FN)

###end of table 1






#att=list(dfN.drop(dfN.columns[-1],axis=1).columns.values)
#print(att)

learner, weights=AdaBoost(dcsnTreeRoot,dfN,att.copy(),dfN,0,1,classPrint,dcsnTreeNodeClass,20)


predictionArr=learnersAggregation(learner,weights,dfN)

print("===========AdaBoost Result on Train =========")

TP,FP,TN,FN=perf_measure(Yaw,predictionArr)
#print(TP,FP,TN,FN)
printMeasure(TP,FP,TN,FN)


predictionArr=learnersAggregation(learner,weights,dfTestN)
#my_set = set(predictionArr)
#my_new_list = list(my_set)
#print(my_new_list)


print("===========AdaBoost Result on Test =========")

TP,FP,TN,FN=perf_measure(testOutput,predictionArr)
#print(TP,FP,TN,FN)
printMeasure(TP,FP,TN,FN)








