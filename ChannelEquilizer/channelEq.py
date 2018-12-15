# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 18:37:40 2018

@author: Asus
"""

import numpy as np

with open('tre.txt') as f:
    content = f.read().split("\n")
    #content=np.array(content[0],dtype=float)
    #print(type(float(content[0][3])))

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
   
class channelClass:
    def __init__(self):
        inputs=loadfile('input.txt')
        self.h = np.array(inputs[0],dtype=float)
        self.n=len(self.h)
        self.mu=float(inputs[1][0])
        self.var=np.array(inputs[1][1],dtype=float)
        
    def distortedOutput(self,contents):
        lst =[]
        for i in range(len(contents[0])-1):
            ans=float(contents[0][i+1])*self.h[0]+float(contents[0][i])*self.h[1]\
            +np.random.normal(self.mu,self.var)
            lst.append(ans)
        return lst    

model=channelClass()
xvectors=model.distortedOutput(content)
print(model.var)


l=2



dictionary={}
for i in range(np.power(l+1,2)-1):
    dictionary[i]=[]
    

for i in range(l,len(content[0])):
    bs=''
    for j in range(0,l+1):
        #print('===='+str(j))
        bs+=content[0][i-j]
    #bs=bs[::-1] 
    #print(bs)
    clss=int(bs,2)
    print(clss)
    xv=[]
    for k in range(0,l):
        xv.append(xvectors[i-l+k])
        #print(xvectors[i-l+k])
    dictionary[clss].append(xv)

#print(int('111',2))






















