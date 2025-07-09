import numpy as np
#--Data Preprocessing Functions--
def z_score_normalize(data_col):
    min_data = min(data_col)
    max_data = max(data_col)
    return (data_col - min_data) / (max_data - min_data)
#--Linear Regression Functions--
class LinearRegression():
    def __init__(self):
        self.w = []
        self.b = 0
    def calculate_gradients(self, data, label):  # self param is the Linear Regression model
        predicts = (np.dot(data, self.w) + self.b).reshape(-1,1)
        label = label.reshape(-1,1)
        errors = (predicts - label).T 
        gradients = 2 *  np.dot(errors, data)
        return [np.array(gradients) / data.shape[0], np.sum(errors) / data.shape[0]]
    def mse_loss(self,predictions,truth):
        predictions = np.array(predictions).flatten()
        truth = np.array(truth).flatten()
        mse = np.sum((np.square(predictions-truth)))/(2*len(predictions))
        return mse
    def batch_gradient_descent_train(self,x,y,epochs=100,eta = 0.3):
        x = x.to_numpy()
        y = y.to_numpy()
        # Initialization
        self.w = np.random.randn(x.shape[1],1)*0.01
        self.b = 1
        for i in range(epochs):
            dw,db = self.calculate_gradients(x,y)
            self.w -= eta * dw.T
            self.b -= eta * db
            predictions = self.predict(x)
            if i % 100 == 0:
                print('Training MSE Loss: ', self.mse_loss(predictions,y))
            elif i == epochs - 1:
                print('Final training MSE Loss: ', self.mse_loss(predictions,y))
            del dw,db
    def predict(self,x):
        return np.dot(x, self.w)  + self.b
#--Evaluation Functions--
def calculate_mae(x,y):
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    return np.mean(np.abs(x-y))
