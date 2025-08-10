import numpy as np
#--Data Preprocessing Functions--
def z_score_normalize(data_col):
    min_data = min(data_col)
    max_data = max(data_col)
    return (data_col - min_data) / (max_data - min_data)
#--Logistic Regression Functions--
class LogisticRegression():
    def __init__(self):
        self.w = []
        self.b = 0
    def sigmoid(self,x):
        predictions = np.dot(x,self.w) + self.b
        e_term = (np.e)**(-predictions)
        return 1 / (1 + e_term)
    def calculate_gradients(self, data, label): 
        predicts = self.predict(data)
        label = label.reshape(-1,1)
        errors = (predicts - label).T 
        gradients = np.dot(errors, data)
        return [np.array(gradients) / data.shape[0], np.sum(errors) / data.shape[0]]
    def mle_log_loss(self,prop,labels):
        prop = np.array(prop).flatten()
        labels = np.array(labels).flatten()
        mle_log_loss = np.mean(-(labels*(np.log(prop)) + (1-labels)*(np.log((1-prop)))))
        return mle_log_loss
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
                print('Training MLE Log Loss, Epoch ',i, ': ', self.mle_log_loss(predictions,y))
            elif i == epochs - 1:
                print('Final training MLE Log Loss: ', self.mle_log_loss(predictions,y))
            del dw,db
    def predict(self,x,label=False):
        x = np.array(x)
        prediction = (self.sigmoid(x)).reshape(-1,1)
        if label == True:
            l = [1 if x >= 0.5 else 0 for x in prediction]
            return l
        else: 
            return prediction
#--Evaluation Functions--
def calculate_precision(x,y):
    TP = 0
    FP = 0
    x = np.asarray(x)
    y = np.asarray(y)
    for i in range(len(x)):
        if y[i] == 1 and x[i] == 1: 
                TP += 1
        if y[i] == 0 and x[i] == 1:
                FP += 1
    return TP / ( TP + FP )
