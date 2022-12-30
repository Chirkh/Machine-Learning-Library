

'''Neural Network from scratch'''
import numpy as np

class connected_Layer:
    ''' Class for a fully connected layer'''
    
    def __init__(self, in_dim, out_dim):
        self.weights=np.random.rand(in_dim, out_dim)-0.5 # centre the weights on 0
       
        self.biases=np.random.rand(1, out_dim)-0.5 # centre the biases on 0
        
    def forward_prop(self, x):
        # Forward propagation
        self.x=x
        self.y=np.dot(self.x, self.weights)+self.biases
        return self.y
    
    def back_prop(self, y_E, eta):
        #backward propgation
        x_E=y_E@(np.transpose(self.weights))
        w_E=np.transpose(self.x)@y_E
        b_E=y_E
        self.weights-=eta*w_E
        self.biases-=eta*b_E
        return x_E
    

class activate_Layer():
    ''' Class to apply activation functions to the layers'''
    def __init__(self, f):
        # Where f is the activation function
        Activations={'relu':self.relu, 'tanh':self.tanh}
        self.f=Activations[f]
        
    def forward_prop(self, x):
        self.x=x
        self.y=self.f(self.x)[0]
        return self.y
    
    def back_prop(self, y_E, eta):
        
        return self.f(self.x)[1]*y_E # Hadamard product (elementwise product)
        
                
    def tanh(self, x):
        '''Returns list with first item as activation and second as derivative'''
        return [np.tanh(x), 1-np.tanh(x)**2]

    def relu(self,x):
        deriv=x.copy()
        deriv[deriv<=0]=0
        deriv[deriv>0]=1
        return [np.maximum(x,0), deriv]


class NN:
    ''' CLass to create the neural network'''
    def __init__(self, loss):
        '''Whenever you initialise a network you initialise with the loss function'''
        Losses={'MSE':self.MSE}
        self.layers=[]
        self.loss=Losses[loss]
        
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def fit(self, x_train, y_train, epochs, eta):
        n=len(x_train)
        y_E=0
        for i in range(epochs):
            for j in range(n):
                y=x_train[j]
                for l in self.layers:
                    y=l.forward_prop(y)
                y_E+=self.loss(y_train[j],y)[0]
                
                error=self.loss(y_train[j], y)[1] # stores dE/dY
                
                for k in self.layers[::-1]:
                    error=k.back_prop(error, eta)
            y_E=y_E/n
            
            print('Epoch :', i+1)
            print('Error: ', y_E)


    def MSE(self, y1,y2):
        '''Returns mean squared error and the derivative dE/dy as well'''
        return [np.mean(np.power((y1-y2),2))/2, (y2-y1)/y1.size]
            
        def predict(self, x):
            pred=[]
            
            for i in range(len(x)):
                y=x[i]
                for i in self.layers:
                    y=i.forward_prop(y)
                pred.append(y)
                
            return pred

'''          
# Now we can test this code on simple example of xor gate
x_train=np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train=np.array([[[0]],[[1]],[[1]],[[0]]])
#print(x_train.shape)

model=NN('MSE')
model.add_layer(connected_Layer(2,3))
model.add_layer(activate_Layer('relu')) 
model.add_layer(connected_Layer(3,1)) 
model.add_layer(activate_Layer('tanh'))  

model.fit(x_train, y_train, epochs=200, eta=0.2)    
'''  
    
                    
                
                    
                
        