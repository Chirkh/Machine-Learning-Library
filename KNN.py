

import numpy as np

class KNeighbours:
    
    def __init__(self, data, k):
        '''
        k : Integer value for the k nearest neighbours
        data: np array of training set, containing features and 
        classifications'''
        
        self.k=k
        self.x_train=data[:,:-1]
        self.y_train=data[:,-1]
        
    @staticmethod
    def Euclidean_dist(x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def predict(self, x):
        '''
        x is the data we want to predict on'''
        y_pred=[self.predict_item(i) for i in x]
        return np.array(y_pred)
    
    def predict_item(self,x):
        dist=[self.Euclidean_dist(x, x2) for x2 in self.x_train]
        '''We want to sort the distances and find indexes of k closest
        x training points and choose their most common label'''
        ordered=np.argsort(dist)
        neighbour_ind=ordered[:self.k] # Find k closest indices
        labels=[self.y_train[i] for i in neighbour_ind]
        major_neighbour=self.majority_label(labels)
        return major_neighbour # Return class which is most common in k nearest labels
    
    def majority_label(self, labels):
        # labels: list
        # Find label occuring most often in k nearest neighbours
        amounts={}
        for i in labels:
            if i not in amounts:
                amounts[i]=1
            else:
                amounts[i]+=1
        maximum=0
        major_neighbour=None
        for key in amounts:
            if amounts[key]>maximum:
                major_neighbour=key
                maximum=amounts[key]
        return major_neighbour
                
'''
#Simple testing example

    
x=[0,0,0,1,1]
y=[0,0,0,1,1]
x=np.reshape(x, (5,1))
y=np.reshape(y,(5,1))
data=np.append(x, y, axis=1)

model=KNeighbours(data,3)
print(model.predict(np.reshape([1], (1,1))))
'''
        
        
