import numpy as np

'''
Dense, fully connected neural network.
These classes take in seperate x and y values rather than accepting a full data array to fit the model like rest of the the 
algorithms. The reason for this is for neural networks you often break the data into training and testing sets and as such a
seperation of x and y values is often useful.
'''

class connected_Layer:
    
    def __init__(self, in_dim, out_dim, f):
        Activations={'relu':self.relu, 'tanh':self.tanh}
        self.f=Activations[f]
        self.weights=np.random.rand(in_dim, out_dim)-0.5
        self.biases=np.random.rand(1, out_dim)-0.5
        print(type(self.biases))
        
    def forward_prop(self, x):
        self.x=x # Entering the layer
        self.y1=np.dot(self.x, self.weights)+self.biases # linear product with weight matrix
        self.y=self.f(self.y1)[0] # Apply the activation to output of linear products
        return self.y
    
    def back_prop(self, y_E, eta):
        ''' Applying backprop equations'''
        x_E_i=self.f(self.y1)[1]*y_E # Intermediate error in x, activation func contribution
        x_E=x_E_i@(np.transpose(self.weights))
        w_E=np.transpose(self.x)@x_E_i 
        b_E=y_E
        self.weights-=eta*w_E
        self.biases-=eta*b_E
        return x_E
    
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
model.add_layer(connected_Layer(2,5, 'relu'))
model.add_layer(connected_Layer(5,1, 'tanh')) 
  

model.fit(x_train, y_train, epochs=400, eta=0.1)
print(model.predict(np.array([[0,0]])))
'''
