import numpy as np
def sigmoid(x,others):
    tuso = np.e**(x)
    mauso = np.sum(np.e**(others))
    return tuso / mauso
def reLU(x):
    return max(0,x)
class SoftmaxRegression():
    def __init__(self):
        self.relu_layer = []
        self.linear_layer = [[[0]*64]*20] # Linear layer representing y = w * x + b
    def predict(self,data): 
        data = [[data]*10]
        for i in range(len(data)):
            a_1 = reLU(data[i])
        a_1 = np.dot(data,self.linear_layer)
    def cross_entropy_loss(self,x):
        pass
    def calculate_gradients(self):
        pass
    def batch_gd_training(self,x):
        pass
    