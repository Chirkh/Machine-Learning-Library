
''' Logistic regression for 2 class classification'''

import numpy as np

class LogisticRegress:
    
    def __init__(self, data):
        # data: numpy array
        self.samples=data.shape[0]
        self.features=data.shape[1] # We keep this to consider the additional bias term as well
        normal_features=self.normalise_features(data[:,:-1])
        self.x=normal_features
        self.x=np.hstack((np.ones((self.samples,1)), self.x))
        self.y=np.reshape(data[:,-1], (self.samples,1))
        self.theta=np.zeros((self.features,1))
        
    
    def normalise_features(self, xs):
        # Normalising all features
        self.mean=np.mean(xs, axis=0)
        self.sigma=np.std(xs, axis=0)
        if self.sigma==0: # To prevent dividing by zero errors
            self.sigma=0.1
        Z=(xs-self.mean)/self.sigma
        return Z
    

    
    def cost(self):
        # Calculating the cost for given theta
        pred=self.x@self.theta
        cost_0=self.y.T@(np.log(self.sigmoid(pred)))
        cost_1=(1-self.y).T@(np.log(1-self.sigmoid(pred)))
        tot_cost=-(cost_0+cost_1)/self.y.shape[0]
        return tot_cost
    
    def grad_desc_fit(self, eta, epochs):
        # Gradient descent
        initial_cost=self.cost()
        print('Initial cost: ', initial_cost)
        for i in range(epochs):
            x_pred=np.matmul(self.x,self.theta)
            grad=np.matmul(self.x.T,(self.sigmoid(x_pred)-self.y))
            self.theta-=grad*eta/len(self.y)
        final_cost=self.cost()
        print('Final cost: ', final_cost)
            
    def predict(self):
        # Predict on fiiting data
        pred=self.x.dot(self.theta)
        pred=self.assign_class(pred)
        return pred
    
    def regress(self, x):
        # Regress on new data
        x=np.hstack((np.ones((len(x),1)), x)) # Prepare data by adding bias terms
        pred=x.dot(self.theta)
        pred=self.assign_class(pred)
        return pred
    
    def assign_class(self, pred):
        # Assign label to predictions from regression
        for i in range(len(pred)):
            if pred[i][0]>0.5:
                pred[i][0]=1
            else:
                pred[i][0]=0
        return pred
        
    
    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))    

'''
Testing on simple data set

x=[0,0,0,1,1]
y=[0,0,0,1,1]
x=np.reshape(x, (5,1))
y=np.reshape(y,(5,1))
data=np.append(x, y, axis=1)

regressor=LogisticRegress(data)
regressor.grad_desc_fit(eta=0.1, epochs=500)
print(regressor.regress(np.reshape([1,0],(2,1))))    
    
 '''       
        
