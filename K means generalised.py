

'''Kmeans algorithm, unsupervised learning'''

import numpy as np

class KMeansCluster:
    '''
    K means unsupervised clustering algorithm, we assume nd features in this
    code
    '''
    def __init__(self, x, k):
        # For ease we assign centroids to positions of some random x points
        self.centroids=np.random.choice(self.x,k) # An array kxn
        self.n=len(x[0]) # number of features, nd space
        self.k=k
        self.x=x
        self.categories=[0 for i in range(len(x))]
        self.prev_categories=[-1 for i in range(len(x))]
        self.dist=[[-1 for i in range(self.k)] for j in range(len(x))]

       
    def cluster(self, x):
        '''We continue the algorithm until convergence'''
        while self.prev_categories!=self.categories:
            self.pre_categories=self.categories.copy()
            self.reassign()
            self.update()
        return self.centroids, self.categories, self.dist
   
   
    def reassign(self):
        # Reassign all data points to nearest cluster
        for i in range(len(self.x)):
            for j in range(self.k):
                self.dist[i][j]=self.euclidean(self.x[i],self.centroids[j])
        for i in range(len(self.dist)):
            self.categories[i]=self.dist[i].index(min(self.dist[i]))+1
           
    def update(self):
        # Update the positions of each centroid after each iteration
        self.centroids=[[0 for i in range(self.n)] for i in range(self.k)]
        for c in range(self.k):
            assign=[self.x[i] for i in range(len(self.x)) if self.categories[i]==c+1] #for every centroid make a list of assignment
            tot=[0 for i in range(self.n)]
            if len(assign)>0:
                for item in assign:
                    for q in len(item):
                        tot[q]+=item[q]
                       
            self.centroids[c]=tot/len(assign)
       
    @staticmethod      
    def euclidean(x, y):
        return ((x[0]-y[0])**2-(x[1]-y[1])**2)**0.5