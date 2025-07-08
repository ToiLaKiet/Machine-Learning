import numpy as np
def compute_cost(data):
    pass

def z_score_normalize(data_col):
    min_data = min(data_col)
    max_data = max(data_col)
    return (data_col - min_data) / (max_data - min_data)
