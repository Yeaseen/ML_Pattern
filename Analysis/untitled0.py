#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:57:27 2018

@author: yeaseen
"""


import matplotlib.pyplot as plt
class run:
    def __init__(self, name):
        self.name = name
    

def sumI(arg1,arg2,arg3,arg4):
    p=arg2
    print(p.name)
    return (arg1+arg3)

def sumII(arg3,arg4):
    return (arg3+arg4)

def mainF(func, arg1, arg2, arg3, arg4,func2,arg5,arg6,classReal,t):
    x=sumI(arg1,arg2,arg3,arg4)
    y=sumII(arg5,arg6)
    return x,y,t

t=run('itsme')

print(mainF(sumI,2,t,3,4,sumII,5,6,run,9))











def plot_histogram(x):
    plt.hist(x, color='gray', alpha=0.5)
    plt.title("Histogram of '{var_name}'".format(var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
#print(X['age'].value_counts())

#plot_histogram(X['age'])

def plot_histogram_dv(x,y):
    plt.hist(list(x[y==0]), alpha=0.5, label='Outcome=0')
    plt.hist(list(x[y==1]), alpha=0.5, label='Outcome=1')
    plt.title("Histogram of '{var_name}' by Outcome Category".format(var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.show()

#plot_histogram_dv(X['age'], Y)