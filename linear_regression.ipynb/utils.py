import numpy as np
#--Data Preprocessing Functions--
def z_score_normalize(data_col):
    min_data = min(data_col)
    max_data = max(data_col)
    return (data_col - min_data) / (max_data - min_data)
#--Linear Regression Functions--
def calculate_gradients(self, data, label):  # self param is the Linear Regression model
    predicts = (np.dot(data, self.w) + self.b).reshape(-1,1)
    label = label.reshape(-1,1)
    errors = (predicts - label).T 
    gradients = np.dot(errors, data)
    print(gradients.shape,gradients)
    return [ np.array(gradients) / data.shape[0], np.sum(errors) / data.shape[0] ]

class LinearRegression():
    def __init__(self):
        self.w = []
        self.b = 0
    def batch_gradient_descent_train(self,x,y,epochs=100,eta = 0.03):
        x = x.to_numpy()
        y = y.to_numpy()
        # Initialization
        self.w = (np.ones(x.shape[1])).reshape(-1,1)
        self.b = 1
        for i in range(epochs):
            gradients = calculate_gradients(self,x,y)
            self.w -= eta * (gradients[0]).T
            self.b -= eta * gradients[1]
            gradients = []
    def predict(self,x):
        x = x.reshape(-1,1)
        return np.dot(x.T, self.w)  + self.b
#--Evaluation Functions--
def calculate_mae(x,y):
    return np.mean(np.abs(x-y))
