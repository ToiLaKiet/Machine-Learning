import numpy as np
EPS = 1e-12
# Activation Functions
def softmax(z): 
    # z: (k, 1) or (k, m). Implemented column-wise.
    z_shift = z - np.max(z, axis=0, keepdims=True)
    expz = np.exp(z_shift)
    return np.array(expz / (np.sum(expz, axis=0, keepdims=True) + EPS)).reshape(-1,1)
def reLU(x):
    return np.array(np.maximum(x,0)).reshape(-1,1)
def relu_derivative(z):
    return (z > 0).astype(float)
def xavier_init(in_dim, out_dim):
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size=(out_dim, in_dim))
# Fully-connected layers' Functions
class DenseLayer():
    def __init__(self, in_dim, out_dim, activation = "softmax"):
        self.w = xavier_init(in_dim, out_dim)
        self.b = np.zeros((out_dim,1))
        self.activation = activation
        self.shapes = [out_dim, in_dim]
    def forward_prop(self, data):
        if self.activation == "relu":
            return np.dot(np.array(data).T, self.w.T) + self.b.T
        elif self.activation == "softmax":
            return np.dot(np.array(data), self.w.T) + self.b.T
    def backward_prop(self, delta_from_next_layer):
        # If this is the output layer (softmax)
        if self.activation == "softmax":
            # delta_from_next_layer is (predicted - actual)
            self.dw = np.dot(delta_from_next_layer, self.input.T)
            self.db = np.sum(delta_from_next_layer, axis=1, keepdims=True)
            return self.dw, self.db, delta_from_next_layer # For the prev layer, the delta is just this one

        # If this is a hidden layer (ReLU)
        elif self.activation == "relu":
            # Need the original Z before ReLU for its derivative
            relu_prime = relu_derivative(self.z)
            # Calculate delta for this layer
            delta = np.dot(self.w.T, delta_from_next_layer) * relu_prime
            self.dw = np.dot(delta, self.input.T)
            self.db = np.sum(delta, axis=1, keepdims=True)
            return self.dw, self.db, delta # Return delta for previous layer

    def calculate_gradient(self, diff, h, w_2, data, z=None):
        '''
        diff : lỗi từ layer sau truyền về (δ)
        h    : đầu ra của hidden layer (sau ReLU)
        w_2  : trọng số layer sau (chỉ dùng khi backprop qua hidden)
        data : input của layer hiện tại
        z    : giá trị trước khi qua activation (cần cho ReLU)
        '''
        activation = self.activation
        if activation == "softmax":
            # δ = ŷ - y
            w_gradient_matrix = np.outer(diff, h)   # (k, m)
            b_gradient_array = diff                 # (k,)
        elif activation == "relu":
            # δ1 = (W2.T δ2) * ReLU'(z)
            relu_grad = relu_derivative(z)       # đạo hàm ReLU
            delta1 = np.dot(w_2.T, diff) * relu_grad
            w_gradient_matrix = np.dot(delta1, data.T)
            b_gradient_array = delta1                   # (m,)
        return w_gradient_matrix, b_gradient_array


# Softmax Regression Functions
class SoftmaxRegression():
    def __init__(self):
        self.relu_layer = DenseLayer(in_dim= 64, out_dim= 25, activation= "relu")
        self.softmax_layer = DenseLayer(in_dim= 25, out_dim= 10, activation= "softmax")
    def predict(self, data): 
        '''
        data : an input picture 
        ---
        return a list of propability that the input could be of that class
        '''
        a_1 = reLU(self.relu_layer.forward_prop(data))
        a_2 = softmax(self.softmax_layer.forward_prop(a_1))
        return a_1, a_2
    def forward(self, data):
        # Store input for the first layer's backprop
        self.input_data = data
        self.hidden_output = self.relu_layer.forward_prop(data)
        self.output_probs = self.softmax_layer.forward_prop(self.hidden_output)
        return self.output_probs
    def cross_entropy_loss(self, props, labels):
        '''   
        props : To each data row, there is a propability to the ground truth label, this props array is the collection of the model's predicted proability for that ground truth
        labels : A matrix for one hot encoded version of the ground truth labels.
        ---
        return a list of CE Loss for all classes
        '''
        return -np.sum(labels * np.log(props + EPS))
    # def calculate_gradients(self, data, label):
    #     '''
    #     data: 1 training image sample
    #     label: one-hot encoded version of that image class
    #     ---- 
    #     return: a dictionary of gradients of all layers
    #     '''
    #     layers = [self.relu_layer, self.softmax_layer]
    #     z_1, z_2 = self.predict(data)
    #     z_1_no_relu = self.relu_layer.forward_prop(data)
    #     diff = z_2 - np.array(label).reshape(-1,1)
    #     # Gradient dictionaries
    #     w1_b1_gradients = {'w': None, 'b': None}
    #     w2_b2_gradients = {'w': None, 'b': None}
    #     w_b = [w1_b1_gradients, w2_b2_gradients]
    #     # Tính gradient từng layer
    #     # Layer 2 (Softmax)
    #     w2, b2 = layers[1].calculate_gradient(
    #         diff= diff, 
    #         h= z_1, 
    #         w_2= None,          # không cần w_2 ở output layer
    #         data= None, 
    #         z= None
    #     )
    #     w_b[1]['w'], w_b[1]['b'] = w2, b2

    #     # Layer 1 (ReLU hidden)
    #     w1, b1 = layers[0].calculate_gradient(
    #         diff= diff, 
    #         h= z_1, 
    #         w_2= self.softmax_layer.w,  # dùng W2 để backprop
    #         data= data, 
    #         z= z_1_no_relu
    #     )
    #     w_b[0]['w'], w_b[0]['b'] = w1, b1
        
    #     return w_b
    def forward(self, data):
        # Store input for the first layer's backprop
        self.input_data = data
        self.hidden_output = self.relu_layer.forward_prop(data)
        self.output_probs = self.softmax_layer.forward_prop(self.hidden_output)
        return self.output_probs

    def backward(self, output_probs, labels):
        # Calculate initial delta for the output layer
        delta_output = output_probs - labels.reshape(-1,1) # Assuming labels are one-hot encoded

        # Backpropagate through Softmax Layer
        dw2, db2, delta_hidden = self.softmax_layer.backward_prop(delta_output)

        # Backpropagate through ReLU Layer
        # Need the original input to the relu_layer
        dw1, db1, _ = self.relu_layer.backward_prop(delta_hidden) # The last delta isn't needed

        return dw1, db1, dw2, db2
    # def stochastic_gd_training(self, data, labels, eta = 0.3, epochs = 100):
    #     for i in range(epochs):
    #         predictions = []
    #         categorical_cel = []
    #         for j in range(data.shape[0]):
    #             w_b = self.calculate_gradients(np.array(data[j]).reshape(-1,1),labels[j])
    #             self.relu_layer.w -= eta * w_b[0]['w']
    #             self.relu_layer.b -= eta * w_b[0]['b']
    #             self.softmax_layer.w -= eta * w_b[1]['w']
    #             self.softmax_layer.b -= eta * w_b[1]['b']
    #             predictions = self.predict(data[j])[1]
    #             categorical_cel.append(self.cross_entropy_loss(props=predictions,labels=labels[j]))
    #             del predictions
    #         epoch_cel = np.mean(categorical_cel)
    #         del categorical_cel
    #         print('Training Epoch: ',i, 'Categorical Cross Entropy Loss: ', epoch_cel)
    def stochastic_gd_training(self, data, labels, eta = 0.3, epochs = 100):
        num_samples = data.shape[0]

        for i in range(epochs):
            epoch_loss = []
            for j in range(num_samples):
                current_data = data[j].reshape(-1, 1) # Ensure (features, 1)
                current_label = labels[j].reshape(-1, 1) # Ensure (num_classes, 1)

                # Forward pass
                output_probs = self.forward(current_data)

                # Calculate loss
                loss = self.cross_entropy_loss(props=output_probs, labels=current_label)
                epoch_loss.append(loss)

                # Backward pass & Gradient calculation
                dw1, db1, dw2, db2 = self.backward(output_probs, current_label)

                # Update weights and biases
                self.relu_layer.w -= eta * dw1
                self.relu_layer.b -= eta * db1
                self.softmax_layer.w -= eta * dw2
                self.softmax_layer.b -= eta * db2

            avg_epoch_loss = np.mean(epoch_loss)
            print(f'Training Epoch: {i+1}, Categorical Cross Entropy Loss: {avg_epoch_loss:.4f}')
