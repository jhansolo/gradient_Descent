# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 15:39:16 2019
for testing the bath gradient descent regressor class created in GD.py
@author: jh
"""
import numpy as np
from sklearn.datasets.samples_generator import make_regression
from sklearn.model_selection import train_test_split
import GD
import time

"""generated testing data"""
N_SAMPLE=1000
N_DIM=1
x, y = make_regression(n_samples=N_SAMPLE, n_features=N_DIM, 
                       n_informative=100, random_state=20, noise=1) 

trainX,predX,trainY,predY=train_test_split(x,y,test_size=0.05)

"""range of learning rates to investigate. Note that this range is VERY sensitive
to N_SAMPLEs. in case of errors/exceptions, first try adjust this range below"""

possibleAlpha=list(np.linspace(0.001/N_SAMPLE,0.1/N_SAMPLE,20))
epoch=500                                               #for BGD
#epoch2=10000                                           #for SGD, unused atm
tol=1e-3
t0=time.time()                                          #basic timing

"""creating batch_gradient object"""
regressor=GD.batch_gradient(trainX,trainY,tol,epoch)
regressor.seeAlpha(possibleAlpha)                        #finds and visualizes learning rates
regressor.fit()                                          #fits regressor to data, if parameter blank, uses optimal alpha from previous step, otherwise manaully specify          

regressor.predict(predX)                                 #predicition

#only meaningful for 2D data
if N_DIM==1:
    regressor.plot()

#metrics
print('total iterations: ',regressor.count)
print('r2 score: ',regressor.r2)
print('weight vector: ',regressor.weights)
print(time.time()-t0)




#"""WIP of the stochastic gradient descent regressor"""
##print('=================')
##t0=time.time()
##clf2=GD.stochastic_gradient(x,y,tol,epoch2)
##clf2.fit(1e-5)
##print(clf2.weights)
##print(clf2.r2)
##print(clf2.count)
##
##print(time.time()-t0)
#
