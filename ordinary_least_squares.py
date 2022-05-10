# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:49:31 2022

@author: avidvans3
"""

"OLS"

import numpy as np 
import pandas as pd

import scipy
X=np.arange(0,15,0.1);
Y=1.2*X+np.random.normal(0,1,len(X))
X=np.transpose(X);
# Y=np.transpose(Y);



X_=np.full(len(X),1)

data1={'col1':X_,'col2':X}
X_1=pd.DataFrame(data=data1)
# # X_1=np.transpose(X_1)
# beta=np.linalg.inv(np.transpose(X_1)*X_1)*np.transpose(X_1)*Y

beta=np.linalg.inv(np.transpose(X_1).dot(X_1)).dot(np.transpose(X_1))

beta=beta.dot(Y)