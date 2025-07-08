import numpy as np
#--Data Preprocessing Functions--
def z_score_normalize(data_col):
    min_data = min(data_col)
    max_data = max(data_col)
    return (data_col - min_data) / (max_data - min_data)
#--Linear Regression Functions--
class LinearRegression():
    def __init__(self):
        pass
    def train(self,x,y):
        model = self
        
        
        pass
    pass
    
