
import numpy as np

class LinearRegress:
    '''
    Linear Regression class with l1 and l2 regularisation built in
    and optimisation through gradient descent.
    '''
    
    def __init__(self, data):
        # data: numpy array of dataset
        self.item_no=data.shape[0]
        self.features=data.shape[1]-1 # Number of x features
        
        bias_terms=np.ones((self.item_no,1))
        normal_features=self.normalise_features(data[:,:-1])
        self.x=np.hstack((bias_terms, normal_features)) #Adding column of ones to x array for bias terms
        self.y=np.reshape(data[:,-1], (self.item_no,1))
        self.theta=np.zeros((self.features+1,1)) #+1 to also consider the bias terms
    
    def normalise_features(self, xs):
        self.mean=np.mean(xs, axis=0)
        self.sigma=np.std(xs, axis=0)
        if self.sigma==0: # To prevent dividing by zero errors
            self.sigma=0.1
        Z=(xs-self.mean)/self.sigma
        return Z
    
    def grad_desc_fit(self, eta, epochs, l1=False, l2=False, lmda=0):
        initial_cost=self.cost(lmda, l1, l2)[0][0]
        print('Initial cost: ', initial_cost)
        for i in range(epochs):
            grad=self.x.transpose().dot((self.x.dot(self.theta)-self.y))
            if l1:
                grad[1:,:] +=lmda*np.sign(self.theta[1:,:]) # Not considering the bias term
            elif l2:
                grad[1:,:] +=2*lmda*self.theta[1:,:] # Not considering the bias term
            self.theta-= (eta*grad)/self.x.shape[0]  
        final_cost=self.cost(lmda, l1, l2)[0][0]
        print('Final cost: ', final_cost)
        
    def cost(self, lmba, l1=False, l2=False):
        base_loss=(np.matmul(self.x, self.theta)-self.y).T@(np.matmul(self.x, self.theta)-self.y)/(2*self.y.shape[0])
        if l1:
            base_loss+=lmba*abs(self.theta).sum()
        elif l2:
            base_loss+=lmba*np.square(self.theta).sum()
        
        return base_loss
    
    def regress(self,x):
        x=(x-self.mean)/self.sigma # normalise the data
        bias_terms=np.ones((x.shape[0],1))
        x=np.hstack((bias_terms, x))
        return x@(self.theta)
    
    
'''  
# Now we make an example to test the code, a straight line

x=np.reshape(np.random.randn(50),(50,1))
y=np.reshape(np.random.randn(50),(50,1))
x=[i for i in range(10)]
y=[2*i for i in x]
x=np.reshape(x,(10,1))
y=np.reshape(y,(10,1))


data=np.append(x, y, axis=1)

regressor=LinearRegress(data)
regressor.grad_desc_fit(eta=0.1, epochs=100, l2=True, l1=False, lmda=0)
  
#print(regressor.theta)            
print(regressor.regress(np.reshape([1],(1,1))))
#print(regressor.x)
#print(regressor.y)
'''
  
