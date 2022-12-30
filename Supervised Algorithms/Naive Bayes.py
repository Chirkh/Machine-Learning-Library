# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:12:16 2022

@author: madhu
"""

import numpy as np

class NaiveBayes:
    
    def __init__(self, data):
        ''' data: np array'''
        self.samples=data.shape[0]
        self.x=data[:,:-1] # NumPy array of all features
        self.y=np.reshape(data[:,-1], (self.samples,1))
        self.n=len(np.unique(self.y)) # Number of classes
        self.Mean=np.zeros((self.n, self.x.shape[1])) # store mean for each feature for each class
        self.Var=np.zeros((self.n, self.x.shape[1]))
        self.prob=np.zeros(self.n) # prior probabilities derived from class frequencies
        self.classes=np.unique(self.y)
        
        '''
        The code below calculates the means, variances and prior probabilities of 
        all the classes.
        '''
        
        for i in range(self.n):
            X_Cat=[]
            for j in range(len(self.y)):
                if self.y[j]==self.classes[i]:
                    X_Cat.append(self.x[j])
            X_Cat=np.array(X_Cat)
            self.Mean[i,:]=X_Cat.mean(axis=0) # mean for each feature for a given class
            self.Var[i,:]=X_Cat.var(axis=0) # variance for each feature for a given class
            self.prob[i]=X_Cat.shape[0]/self.samples # prior prob for each class derived from frequencies
            
    def predict(self, x):
        y_pred=[]
        for item in x:
            # For each item we predict the class with highest posterior prob
            # Using log of bayes theorem
            max_post_prob=np.NINF
            class_ind=0
            for i in range(self.n):
                p=np.log(self.prob[i])
                posterior_prob=np.sum(np.log(self.gaussian(i, item)))
                posterior_prob+=p
                if posterior_prob>=max_post_prob:
                    max_post_prob=posterior_prob
                    class_ind=i
            y_pred.append(self.classes[class_ind])
        return np.array(y_pred)
    
    def gaussian(self, i, x):
        # We use a gaussian to model the probabilities p(x_i|y)
        # in the process of calculating the posterior probability
        mu=self.Mean[i]
        sigma2=self.Var[i]
        # If variance is 0, then we simply set it to a small value
        if sigma2==0:
            sigma2=0.001
        return np.exp(-((x-mu)**2/2*sigma2))/(2*np.pi*sigma2)**0.5
    
    

#Testing on simple data set

x=[1,0,2,5,6]
y=[0,0,0,1,1]
x=np.reshape(x, (5,1))
y=np.reshape(y,(5,1))
data=np.append(x, y, axis=1)            
                    
classifier=NaiveBayes(data)
print(classifier.predict(np.array([[7]])))                
                    
        
        