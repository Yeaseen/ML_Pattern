#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 17:48:29 2018

@author: yeaseen
"""

import pandas as pd
import math
from sklearn.model_selection import train_test_split

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



def entropyCalc(x,y):
    p1=x/(x+y)
    p2=y/(x+y)
    if(p1==0 or p2==0):
        res=0
    else:
        res=-p1*math.log2(p1)-p2*math.log2(p2)
    return res 

df = pd.read_csv('creditcard.csv')

#print(df.info())


isFraud = df[df['Class'] == 1]
print(len(isFraud))
isNotFraud= df.iloc[200:5900]
isNotFraud= isNotFraud[isNotFraud['Class'] == 0 ]
print(len(isNotFraud))





frames=[isNotFraud,isFraud]

result= pd.concat(frames)

print(len(result))

#result.to_csv('sampledcreditcard.csv', encoding='utf-8', index=False)

dfFull=dataCleaning(result)



#result_output=dfN.iloc[:-1].values

#print(len(result_output))





dfN,dfTestN= train_test_split(dfFull,test_size=0.2,shuffle=True,)


print(dfN.head(5))

print(dfTestN.head(5))
print(dfN['Class'].value_counts())
print(dfTestN['Class'].value_counts())