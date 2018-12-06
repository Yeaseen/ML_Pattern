import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df

 
def entropyCalc(x,y):
    p1=x/(x+y)
    p2=y/(x+y)
    res=-p1*math.log2(p1)-p2*math.log2(p2)
    return res    



    

df = pd.read_csv('adult_train.csv', na_values=['#NAME?'])
#print(df.head(5))

#print(df['age'].value_counts())

df['income']= [0 if x == '<=50K' else 1 for x in df['income']]
X = df.drop('income', 1)
Y = df.income

total_zeros = Y[Y == 1].count()
total_ones = Y[Y == 0].count()
total_output= total_zeros + total_ones
total_entropy=entropyCalc(total_zeros,total_ones)
print(total_entropy) 


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
#print(Y.value_counts())
#print(X.head(5))
#print(Y.head(5))
#plot_histogram_dv(X['native_country'], Y)
#print(Y)
'''
for col_name in X.columns:
    if X[col_name].dtypes == 'object':
        unique_cat = len(X[col_name].unique())
        print(col_name)
        print("Feature '{col_name}' has {unique_cat} unique categories\
              ".format(col_name=col_name, unique_cat=unique_cat))
    elif X[col_name].dtypes == 'int64':
        unique_value = len(X[col_name].unique())
        print("     Feature '{col_name}' has {unique_value} unique values\
              ".format(col_name=col_name, unique_value=unique_value)) 
        

print(X['native_country'].value_counts().sort_values(ascending=False).head(10))
'''

#print(df['native_country'].value_counts())
#print(X['native_country'].value_counts())
X['native_country'] = ['United-States' if x == 'United-States' else 'Other' for x in X['native_country']]

#print(X['native_country'].value_counts().sort_values(ascending=False))
#plot_histogram_dv(X['native_country'], Y)   

dummy_list=[]

for col_name in X.columns:
    if X[col_name].dtypes == 'object':
        unique_cat = len(X[col_name].unique())
        dummy_list.append(col_name)

#print(dummy_list)

X = dummy_df(X, dummy_list)
#print(X.head(5))


#print(X.isnull().sum().sort_values(ascending=False).head())


imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(X)
X = pd.DataFrame(data=imp.transform(X) , columns=X.columns)

#print(X.isnull().sum().sort_values(ascending=False).head())
#print(X.head(5))
'''
 
'''
#print(X['native_country_United-States'].value_counts())
print(Y.dtypes)        


#
#print(ff)


for col_name in X.columns:
    unique_cat = len(X[col_name].unique())
    if(unique_cat > 2):
        print(col_name)
        ff=np.mean(X[col_name])
        print(ff)
        X[col_name]= [ 0.0 if x <= ff else 1.0 for x in X[col_name]]
        print(X[col_name].value_counts())

        
   
'''
for col_name in X.columns:
        unique_cat = len(X[col_name].unique())
        print(col_name)
        print("Feature '{col_name}' has {unique_cat} unique categories\
              ".format(col_name=col_name, unique_cat=unique_cat))
   ''' 



#print(Z.value_counts())

#binarization(Z)
#Z=X.age
#print(Z)
#uniAge=set(Z)
#uniAge=X['fnlwgt'].unique()
#uniAge.sort()
#print(np.mean(uniAge))
#print(len(uniAge))

#print(entropyCalc(7,17))





#X.age=Z
#print(X['age'].value_counts())


print('Hello')
