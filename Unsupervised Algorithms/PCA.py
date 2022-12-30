

import numpy as np

class PCA:
    ''' Implements PCA as a method to reuce the dimensionality of the data'''
    def __init__(self, x, k):
        ''' k: How many dimensions we want to reduce to
            x: np array of features
        '''
        self.k=k
        self.x=x
        self.components=None
       
        self.mean=np.mean(x, axis=0)
        self.x-=self.mean # Centering the data before we calculate cov matrix
       
        covariance=np.cov(self.x.T)
        v,lmbda=np.linalg.eig(covariance) # Calculate the eigenvectors and eigenvalues
        v=v.T # We transpose the eigenvectors so they are easier to work with, eogenvectors now go across columns
        # so c[i] is the eigenvector correspnding to eiganvalue [i]
        # We want the eigenvectors corresponding to k highest eigenvalues
        indices=np.argsort(lmbda)
        k_indices=indices[::-1][:self.k]
        eigenvectors=v[k_indices]
        self.components=eigenvectors
       
    def apply(self,x):
        x-=self.mean
        return np.dot(x, self.components.T) # We make the projections of the data on the eigenvector components