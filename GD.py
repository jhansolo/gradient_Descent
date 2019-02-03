# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 11:33:36 2019

an exercise in OOP class creation/inheritance and gradient descent by creating a 
creating a batch_gradient regressor class to implement the 
simple naive batch gradient descent on, using the
linear regression using the LMS update rule as outlined
in the CS229 notes from Andrew Ng.

the batch_gradient has several main methods, most useful of which are:
    
    1. seeAlpha: visualizes the change in loss function as a result of different
    learning rates (alpha) chosen from a user specified range. Determines the 
    optimal rate for subsequent fitting (with the option of overriding) and 
    predictions
    
    2. fit: finds the weight vectors describing the linear regression of the dataset.
    if no parameter is passed, learning rate alpha is the optimal alpha found from
    the seeAlpha method
    
    3. plot: for 2-dimensional data, visualizes the training data, MSE over iterations,
    and effect of different learning rates. for >2D data, not called 

NOTE that the X array must be shape (n_sample,n_dim-1)
and that the y array must be shape (n_sample,)
see accompanying main.py for example

a basic definition for stochastic gradient descent is also written below. It is
implemented in a stochastic_graident class, which inherited from the batch_gradient
class. Not as fully developed as the batch version at the moment.



@author: jh
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn import metrics
import pandas as pd



def bgd(x,y,epoch,alpha,tol):
    """basic implementation of the batch gd"""
    
    m=len(x)
    trainX = np.c_[ np.ones(m), x] # insert column
    dim=trainX.shape[1]
    trainY=copy.copy(y)
    theta=np.ones(dim)
    lossRecord=[]

    for i in range(0,epoch):
        oldTheta=copy.copy(theta)
        hypothesis=np.dot(trainX,theta)
        loss=hypothesis-trainY
        lossRecord.append(np.asscalar(np.linalg.norm(loss)))
        gradient=np.dot(trainX.T,loss)
        theta=theta-alpha*np.array(gradient)
        diff=np.linalg.norm(theta-oldTheta)
        if diff<tol:
            break
    count=i    
    weights=theta
    history=lossRecord
    regressedY=np.dot(trainX,weights)
    r2=metrics.r2_score(regressedY,trainY)
    return count,weights,history,r2


class batch_gradient:
    """batch gradient regressor class"""
    weights=None 
    history=None
    count=None
    r2=None
    trainX=None
    trainY=None
    alpha=None
    testX=None
    testY=None
    plotAlpha=False
    
    
    def __init__(self,x,y,tol,epoch,):
        self.x=x
        self.y=y
        self.trainX=copy.copy(x)
        self.trainY=copy.copy(y)
        self.tol=tol
        self.epoch=epoch

        
    def fit(self,a=None):
        if a is not None:
            self.alpha=a
            self.plotAlpha=False
        self.count,self.weights,self.history,self.r2=bgd(self.trainX,self.trainY,self.epoch,self.alpha,self.tol)

    def seeAlpha(self,alphaRange):
        self.collection=pd.DataFrame()
        rate=0
        self.optimalAlpha=0
        for i in alphaRange:
            localCount,localWeights,localHistory,localR2=bgd(self.trainX,self.trainY,self.epoch,i,self.tol)            
            self.collection[str(i)]=pd.Series(localHistory)
            drop=localHistory[0]-localHistory[1]
            if drop>rate:
                rate=drop
                self.optimalAlpha=i
        self.alpha=self.optimalAlpha
        self.plotAlpha=True
        print('auto-generated alpha: ',self.optimalAlpha)

        
    def predict(self,newX=None):
        if newX is None:
            self.testX=copy.copy(self.trainX)
        else:
            self.testX=newX
            m= len(self.testX)
            self.testX = np.c_[ np.ones(m), self.testX]
        self.testY=np.array(np.dot(self.testX,self.weights)).flatten('F')

        
    def plot(self,forecast=0):
        
        plt.style.use('seaborn')
        fig=plt.figure()
        grid=plt.GridSpec(2,8)
        fig.set_figheight(6)
        fig.set_figwidth(12)
        dataPlot=fig.add_subplot(grid[0,0:6])
        costPlot=fig.add_subplot(grid[1,0:6])
        alphaPlot=fig.add_subplot(grid[0:2,6:8])
        
        if self.count<self.epoch-1:
            text='linear regression: batch gradient descent after {} iterations reached {} tolerance'.format(self.count, self.tol)
        else:
            text='linear regression: batch gradient descent after {} iterations without reaching {} tolerance'.format(self.count, self.tol)
#
        dataPlot.set_title(text)
        
        dataPlot.scatter(self.x,self.y,c='g',label='training data',s=2)
        dataPlot.scatter(self.testX[:,-1],self.testY,s=10,c='r',label='prediction data')
        
        bigX=np.max(np.concatenate((np.array(self.x).flatten('F'),self.testX[:,-1])))
        smallX=np.min(np.concatenate((np.array(self.x).flatten('F'),self.testX[:,-1])))
        xStart=smallX-forecast
        xEnd=bigX+forecast
        lineX=np.matrix([xStart,xEnd]).T
        lineX2=np.array(lineX).flatten('F')
        lineX=np.c_[np.ones(len(lineX)),lineX]
        lineY=np.dot(lineX,self.weights)
        lineY=np.array(lineY).flatten('F')
        
        dataPlot.plot(lineX2,lineY,c='k',label='regression line')
        dataPlot.text(0.05,0.8,s='regression weight vector = {}, {}. R^2={}'.format(round(self.weights[0],3),round(self.weights[1],3),round(self.r2,3))
                ,transform=dataPlot.transAxes)
        dataPlot.legend(framealpha=0.5, loc=4)
        
        costPlot.plot(self.history)
        bestAlpha=format(self.alpha,'.2e')
        alphaPlot.set_title('MSE vs learning rate')

        if self.plotAlpha==True:
            alphaPlot.plot(self.collection)
            alphaLegends=np.array(self.collection.columns.values,dtype=float)
            sciNo_legend=[]
            for value in alphaLegends:
                sciNo_legend.append(format(value,'.2e'))
            
            alphaPlot.legend(sciNo_legend,prop={'size':10})

        
            alphaText='optimal \nalpha:\n{}'.format(bestAlpha)
            alphaPlot.text(0.3,0.5,s=alphaText,transform=alphaPlot.transAxes)
            costTitle=('MSE over iterations, auto-generated alpha ={}'.format(bestAlpha))
        else:
            costTitle=('MSE over iterations, manually picked alpha ={}'.format(bestAlpha))
        costPlot.set_title(costTitle)

        plt.tight_layout()

"""experimental stochastic gradient class, WIP"""

def update(i,trainX,trainY,theta,alpha):
    oldTheta=copy.copy(theta)
    localX=trainX[i]
    localY=trainY[i]

    hypothesis=np.dot(localX,theta)
    loss=hypothesis-localY
    gradient=loss*localX
    theta=theta-alpha*gradient
    diff=np.linalg.norm(theta-oldTheta)
    return diff, oldTheta, theta, loss

def sgd(x,y,epoch,alpha,tol):
    m=len(x)
    trainX = np.c_[ np.ones(m), x] # insert column
    dim=trainX.shape[1]
    trainY=copy.copy(y)
    theta=np.ones(dim)
    lossRecord=[]

    diff, oldTheta,theta, loss=update(0,trainX,trainY,theta,alpha)
    k=0
    
    while diff>tol and k<epoch:
        for i in range(len(trainX)):
            diff, oldTheta,theta, loss=update(i,trainX,trainY,theta,alpha)
            lossRecord.append(loss)
            k+=1
    weights=theta
    count=k
    history=lossRecord
    regressedY=np.dot(trainX,weights)
    r2=metrics.r2_score(regressedY,trainY)
    return count,weights,history,r2

class stochastic_gradient(batch_gradient):
        
    def fit(self,a=None):
        if a is not None:
            self.alpha=a
            self.plotAlpha=False
        self.count,self.weights,self.history,self.r2=sgd(self.trainX,self.trainY,self.epoch,self.alpha,self.tol)




