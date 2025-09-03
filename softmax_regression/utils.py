import numpy as np
EPS = 1e-12
# Activation Functions
def softmax(z, eps=1e-9): 
    # z: (m,1) vector cột
    z = z.reshape(-1, 1)                        # đảm bảo shape (m,1)
    z_shift = z - np.max(z)                     # tránh overflow
    expz = np.exp(z_shift)
    return expz / (np.sum(expz) + eps)          # vẫn trả về (m,1)

def reLU(x):
    return np.array(np.maximum(x,0)).reshape(-1,1)
def relu_derivative(z):
    return np.array((z > 0).astype(float)).reshape(-1,1)
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
        self.input = data # Store input for backprop
        self.z = np.dot(data.T, self.w.T) + self.b.T # Pre-activation
        if self.activation == "relu":
            self.a = reLU(self.z)
        elif self.activation == "softmax":
            self.a = softmax(self.z)
        return self.a
    def backward_prop(self, delta_from_next_layer, w_from_next_layer=None):
        # If this is the output layer (softmax)
        if self.activation == "softmax":
            self.dw = np.dot(delta_from_next_layer, self.input.T)
            self.db = np.sum(delta_from_next_layer, axis=1, keepdims=True)
            # For the prev layer, the delta is just this one
            return self.dw, self.db, delta_from_next_layer

        # If this is a hidden layer (ReLU)
        elif self.activation == "relu":
            if w_from_next_layer is None:
                raise ValueError("w_from_next_layer must be provided for ReLU backward_prop")
            relu_prime = relu_derivative(self.z)
            # CORRECTED LINE: Use w_from_next_layer

            delta = np.dot(w_from_next_layer.T, delta_from_next_layer) * relu_prime
            self.dw = np.dot(delta, self.input.T)
            self.db = np.sum(delta, axis=1, keepdims=True)
            return self.dw, self.db, delta # Return delta for previous layer
# Softmax Regression Functions
class SoftmaxRegression():
    def __init__(self):
        self.relu_layer = DenseLayer(in_dim= 64, out_dim= 25, activation= "relu")
        self.softmax_layer = DenseLayer(in_dim= 25, out_dim= 10, activation= "softmax")
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
    def backward(self, output_probs, labels):
        delta_output = output_probs - labels.reshape(-1,1)

        # Backpropagate through Softmax Layer
        # softmax_layer.backward_prop returns its delta_output to be used by the previous layer
        dw2, db2, delta_for_hidden_layer = self.softmax_layer.backward_prop(delta_output)

        # Backpropagate through ReLU Layer
        # Pass the weights of the *softmax layer* (next layer relative to ReLU)
        dw1, db1, _ = self.relu_layer.backward_prop(np.array(delta_for_hidden_layer).reshape(-1,1), self.softmax_layer.w)

        return dw1, db1, dw2, db2
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
