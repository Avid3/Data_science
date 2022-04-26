# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 23:31:54 2022

@author: avidvans3
"""

"Random forest classifier"
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

data=pd.read_csv('C:\\Users\\avidvans3\\wine.csv')
data.columns=['Class','Alcohol','Malic Acid','Ash','Alcalinity','Magnesium','Total phenols','Flavanoids','Non flavoniod phenols','Proanthocyanins','Color intensity','Hue','OD280','Proline']
X=data[['Alcohol','Malic Acid','Ash','Alcalinity','Magnesium','Total phenols','Flavanoids','Non flavoniod phenols','Proanthocyanins','Color intensity','Hue','OD280','Proline']]
y=data[['Class']]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(random_state=42,n_jobs=1,n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(Xtrain,ytrain.values.ravel())

w=clf.predict(Xtest)

a=clf.score(Xtest,ytest)