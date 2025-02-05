import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.random.randn(1, self.hidden_size)
        self.bias_output = np.random.randn(1, self.output_size)

    def feedforward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.final_input)
        return self.output

    def backpropagate(self, X, y, learning_rate=0.1):
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            self.feedforward(X)
            self.backpropagate(X, y, learning_rate)
            if epoch % 100 == 0:
                predictions = self.output > 0.5
                accuracy = np.mean(predictions == y) * 100
                loss = np.mean(np.square(y - self.output))
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)
nn.train(X, y, epochs=10000, learning_rate=0.1)

predictions = nn.feedforward(X)
print("Final Output after training:")
print(predictions)

final_accuracy = (predictions > 0.5) == y
accuracy = np.mean(final_accuracy) * 100
print(f'Final Accuracy: {accuracy:.2f}%')
