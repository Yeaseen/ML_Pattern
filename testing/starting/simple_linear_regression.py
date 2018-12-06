

import dataset
import numpy as np


import pandas as pd

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')

#here X is a independent variable. [taking all row, allcolumn except last one]

X= dataset.iloc[:, :-1].values

#here Y is the dependent variable that is Purchased .
Y= dataset.iloc[:, 1].values


#splitting the dataset into training set and test set

#import model_selection instead of cross_validation bcx version changed

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size= 0.8, random_state=0)

#feature scaling: for euclidian distance, if one part is high value and another is low value,
#then, high value dominates over it , so we need to do scaling on both part between 0 to 1

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


# Fitting SLR to the training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, Y_train)