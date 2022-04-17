# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 17:55:25 2022

@author: Amey
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, recall_score, precision_score, average_precision_score, f1_score, classification_report, accuracy_score, plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix
df = pd.read_csv('train(1).csv', na_values='?')
df=df.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
df = df.dropna().copy()


  
# creating an object 
# of the LabelBinarizer
label_binarizer = LabelBinarizer()
  
# fitting the column 
# TEMPERATURE to LabelBinarizer
label_binarizer_output = label_binarizer.fit_transform( df['Sex'])
  
# creating a data frame from the object

# df=df.drop(columns=['Sex'],axis=1)
df = df.dropna().copy()
label_df=label_binarizer_output;

# df=pd.concat([df,label_df],axis=1)
df['Sex']=label_df;
                     
df = pd.get_dummies(df, columns=['Pclass'], drop_first=True)

random_seed = 888
df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_seed, stratify=df['Survived'])

print(df_train.shape)
print(df_test.shape)
print()
print(df_train['Survived'].value_counts(normalize=True))
print()
print(df_test['Survived'].value_counts(normalize=True))


numeric_cols = ['Age']
cat_cols = list(set(df.columns) - set(numeric_cols) - {'Survived'})
cat_cols.sort()

scaler = StandardScaler()
scaler.fit(df_train[numeric_cols])

def get_features_and_target_arrays(df, numeric_cols, cat_cols, scaler):
    X_numeric_scaled = scaler.transform(df[numeric_cols])
    X_categorical = df[cat_cols].to_numpy()
    X = np.hstack((X_categorical, X_numeric_scaled))
    y = df['Survived']
    return X, y


X, y = get_features_and_target_arrays(df_train, numeric_cols, cat_cols, scaler)

clf = LogisticRegression(penalty='none') # logistic regression with no penalty term in the cost function.

clf.fit(X, y)

X_test, y_test = get_features_and_target_arrays(df_test, numeric_cols, cat_cols, scaler)

plot_roc_curve(clf, X_test, y_test)


test_prob = clf.predict_proba(X_test)[:, 1]
test_pred = clf.predict(X_test)
plot_roc_curve(clf, X_test, y_test)
plot_precision_recall_curve(clf, X_test, y_test)


test_prob = clf.predict_proba(X_test)[:, 1]
test_pred = clf.predict(X_test)


print('Log loss = {:.5f}'.format(log_loss(y_test, test_prob)))
print('AUC = {:.5f}'.format(roc_auc_score(y_test, test_prob)))
print('Average Precision = {:.5f}'.format(average_precision_score(y_test, test_prob)))
print('\nUsing 0.5 as threshold:')
print('Accuracy = {:.5f}'.format(accuracy_score(y_test, test_pred)))
print('Precision = {:.5f}'.format(precision_score(y_test, test_pred)))
print('Recall = {:.5f}'.format(recall_score(y_test, test_pred)))
print('F1 score = {:.5f}'.format(f1_score(y_test, test_pred)))

print('\nClassification Report')
print(classification_report(y_test, test_pred))

print('Confusion Matrix')
plot_confusion_matrix(clf, X_test, y_test)
