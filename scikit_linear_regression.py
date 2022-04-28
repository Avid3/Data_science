# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:07:32 2022

@author: avidvans3
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import linear_model

data=pd.read_csv('C:\\Users\\avidvans3\\auto-mpg.csv')
data.columns=['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','orgin','carname']
data=data[['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','orgin']]
data['model year']=data['model year']-data['model year'].min()
data=data.replace({'?':np.nan}).dropna(axis=0).astype(float)

X=data[['cylinders','displacement','horsepower','weight','acceleration','model year','orgin']]
y=data[['mpg']]
# X['horsepower']=pd.to_numeric(X['horsepower']).astype(float)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
clf=linear_model.LinearRegression()
reg=clf.fit(Xtrain,ytrain)
y_test=clf.predict(Xtest)
r_square=reg.score(Xtrain,ytrain)
intercepts_=reg.intercept_
coefs=reg.coef_

print(clf.summary())