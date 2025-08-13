import numpy as np
# Activation Functions
def softmax(x):
    tuso = np.e**(x)
    mauso = np.sum(np.e**(x))
    return tuso / mauso
def reLU(x):
    return max(0,x)
# Fully-connected layers' Functions
class DenseLayer():
    def __init__(self, in_dim, out_dim, activation = "linear"):
        self.w = np.zeros(in_dim)
        self.b = np.zeros(out_dim)
        self.activation = activation
    def forward_prop(self,data):
        if self.activation != "relu":
            z = np.dot(data,self.w) + self.b
        else:
            z = reLU(np.dot(data,self.w) + self.b)
        return z
    def calculate_gradient(self):
        pass
# Softmax Regression Functions
class SoftmaxRegression():
    def __init__(self):
        self.relu_layer = DenseLayer(in_dim=64, out_dim=25, activation="relu")
        self.linear_layer = DenseLayer(in_dim=25, out_dim=10, activation="linear")
    def predict(self,data): 
        data = [ data ] * self.relu_layer.w
        a_1 = self.relu_layer.forward_prop(data)
        a_2 = softmax(self.linear_layer.forward_prop(a_1))
        return a_2
    def cross_entropy_loss(self,props,labels):
        '''   
        props : To each data row, there is a propability to the ground truth label, this props array is the collection of the model's predicted proability for that ground truth
        labels : A matrix for one hot encoded version of the ground truth labels.
        ---
        return a list of CE Loss for all classes
        '''
        props = np.array(props).flatten()
        labels = np.array(labels).flatten()
        CE_loss = np.zeros(10)
        for i in range(labels.shape[0]):
            label = np.argmax(labels[i])
            prop = props[i]
            CE_loss[label] += -(label*(np.log(prop)) + (1-label)*(np.log((1-prop))))
        return np.mean(CE_loss)
    def calculate_gradients(self):
        pass
    def batch_gd_training(self,x):
        pass
    