import numpy as np

class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)
    
    def forward(self, input):
        self.input = input
        self.output = np.zeros((self.num_filters, 
                               input.shape[1] - self.filter_size + 1, 
                               input.shape[2] - self.filter_size + 1))
        
        for i in range(self.num_filters):
            for j in range(self.output.shape[1]):
                for k in range(self.output.shape[2]):
                    self.output[i, j, k] = np.sum(
                        input[:, j:j+self.filter_size, k:k+self.filter_size] * self.filters[i]
                    )
        return self.output

    def backward(self, d_output, learning_rate):
        d_input = np.zeros_like(self.input)
        d_filters = np.zeros_like(self.filters)
        
        for i in range(self.num_filters):
            for j in range(d_output.shape[1]):
                for k in range(d_output.shape[2]):
                    d_input[:, j:j+self.filter_size, k:k+self.filter_size] += \
                        self.filters[i] * d_output[i, j, k]
                    d_filters[i] += self.input[:, j:j+self.filter_size, k:k+self.filter_size] * \
                        d_output[i, j, k]
        
        self.filters -= learning_rate * d_filters
        return d_input

class MaxPoolLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size
    
    def forward(self, input):
        self.input = input
        self.output = np.zeros((input.shape[0],
                               input.shape[1] // self.pool_size,
                               input.shape[2] // self.pool_size))
        
        for i in range(self.output.shape[1]):
            for j in range(self.output.shape[2]):
                self.output[:, i, j] = np.max(
                    input[:, i*self.pool_size:(i+1)*self.pool_size, 
                          j*self.pool_size:(j+1)*self.pool_size],
                    axis=(1, 2)
                )
        return self.output
    
    def backward(self, d_output):
        d_input = np.zeros_like(self.input)
        
        for i in range(d_output.shape[1]):
            for j in range(d_output.shape[2]):
                temp = self.input[:, i*self.pool_size:(i+1)*self.pool_size,
                                 j*self.pool_size:(j+1)*self.pool_size]
                mask = temp == np.max(temp, axis=(1, 2))[:, np.newaxis, np.newaxis]
                d_input[:, i*self.pool_size:(i+1)*self.pool_size,
                        j*self.pool_size:(j+1)*self.pool_size] = mask * d_output[:, i:i+1, j:j+1]
        return d_input

class FCLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.bias = np.zeros(output_size)
    
    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output
    
    def backward(self, d_output, learning_rate):
        d_input = np.dot(d_output, self.weights.T)
        d_weights = np.dot(self.input.T, d_output)
        d_bias = np.sum(d_output, axis=0)
        
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        return d_input

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return x > 0

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Example usage
class CNN:
    def __init__(self):
        self.conv1 = ConvLayer(num_filters=3, filter_size=3)
        self.pool1 = MaxPoolLayer(pool_size=2)
        self.fc1 = FCLayer(input_size=27, output_size=10)  # Adjust input size based on your data
    
    def forward(self, x):
        x = self.conv1.forward(x)
        x = relu(x)
        x = self.pool1.forward(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.fc1.forward(x)
        return softmax(x)
    
    def train(self, x, y, learning_rate=0.01):
        # Forward pass
        conv_output = self.conv1.forward(x)
        relu_output = relu(conv_output)
        pool_output = self.pool1.forward(relu_output)
        flatten_output = pool_output.reshape(pool_output.shape[0], -1)
        fc_output = self.fc1.forward(flatten_output)
        output = softmax(fc_output)
        
        # Backward pass
        d_output = output - y
        d_fc = self.fc1.backward(d_output, learning_rate)
        d_fc = d_fc.reshape(pool_output.shape)
        d_pool = self.pool1.backward(d_fc)
        d_relu = d_pool * relu_derivative(conv_output)
        self.conv1.backward(d_relu, learning_rate)

# Test the CNN
if __name__ == "__main__":
    # Generate dummy data
    X = np.random.randn(1, 28, 28)  # One 28x28 image
    y = np.zeros((1, 10))           # One-hot encoded label
    y[0, 3] = 1                     # Class 3 is the target
    
    cnn = CNN()
    
    # Training loop
    for i in range(100):
        cnn.train(X, y)
        if i % 10 == 0:
            output = cnn.forward(X)
            loss = -np.sum(y * np.log(output))
            print(f"Iteration {i}, Loss: {loss}")