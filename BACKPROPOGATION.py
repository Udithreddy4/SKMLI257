import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Adam optimizer components
def adam_optimizer(m, v, t, grads, beta1=0.9, beta2=0.999, epsilon=1e-8, learning_rate=0.001):
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * grads**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    params_update = -learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return m, v, params_update


data = pd.read_csv("C:\\Users\\himap\\Downloads\\diabetes.csv")

X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, lambda_reg=0.01, patience=100):
        # Initialize weights, biases, and Adam optimizer parameters
        self.weights_input_hidden1 = np.random.randn(input_size, hidden_size1)
        self.weights_hidden1_hidden2 = np.random.randn(hidden_size1, hidden_size2)
        self.weights_hidden2_output = np.random.randn(hidden_size2, output_size)
        self.bias_hidden1 = np.zeros((1, hidden_size1))
        self.bias_hidden2 = np.zeros((1, hidden_size2))
        self.bias_output = np.zeros((1, output_size))

        # Adam optimizer variables
        self.m_input_hidden1, self.v_input_hidden1 = np.zeros_like(self.weights_input_hidden1), np.zeros_like(self.weights_input_hidden1)
        self.m_hidden1_hidden2, self.v_hidden1_hidden2 = np.zeros_like(self.weights_hidden1_hidden2), np.zeros_like(self.weights_hidden1_hidden2)
        self.m_hidden2_output, self.v_hidden2_output = np.zeros_like(self.weights_hidden2_output), np.zeros_like(self.weights_hidden2_output)
        self.m_bias_hidden1, self.v_bias_hidden1 = np.zeros_like(self.bias_hidden1), np.zeros_like(self.bias_hidden1)
        self.m_bias_hidden2, self.v_bias_hidden2 = np.zeros_like(self.bias_hidden2), np.zeros_like(self.bias_hidden2)
        self.m_bias_output, self.v_bias_output = np.zeros_like(self.bias_output), np.zeros_like(self.bias_output)
        
        self.lambda_reg = lambda_reg
        self.losses = []
        self.patience = patience
        self.best_loss = np.inf
        self.early_stop_count = 0

    def forward(self, X):
        self.input_layer = X
        self.hidden_layer1_input = np.dot(self.input_layer, self.weights_input_hidden1) + self.bias_hidden1
        self.hidden_layer1_output = sigmoid(self.hidden_layer1_input)

        self.hidden_layer2_input = np.dot(self.hidden_layer1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        self.hidden_layer2_output = sigmoid(self.hidden_layer2_input)

        self.output_layer_input = np.dot(self.hidden_layer2_output, self.weights_hidden2_output) + self.bias_output
        self.output_layer_output = sigmoid(self.output_layer_input)
        return self.output_layer_output

    def backward(self, X, y, learning_rate, t):
        output_error = self.output_layer_output - y
        output_delta = output_error * sigmoid_derivative(self.output_layer_output)

        hidden2_error = output_delta.dot(self.weights_hidden2_output.T)
        hidden2_delta = hidden2_error * sigmoid_derivative(self.hidden_layer2_output)

        hidden1_error = hidden2_delta.dot(self.weights_hidden1_hidden2.T)
        hidden1_delta = hidden1_error * sigmoid_derivative(self.hidden_layer1_output)

        # L2 Regularization for weights
        regularization_term = self.lambda_reg * (np.sum(self.weights_input_hidden1**2) + np.sum(self.weights_hidden1_hidden2**2) + np.sum(self.weights_hidden2_output**2))

        # Adam optimizer updates
        self.m_input_hidden1, self.v_input_hidden1, dw_input_hidden1 = adam_optimizer(self.m_input_hidden1, self.v_input_hidden1, t, self.input_layer.T.dot(hidden1_delta) + self.lambda_reg * self.weights_input_hidden1)
        self.m_hidden1_hidden2, self.v_hidden1_hidden2, dw_hidden1_hidden2 = adam_optimizer(self.m_hidden1_hidden2, self.v_hidden1_hidden2, t, self.hidden_layer1_output.T.dot(hidden2_delta) + self.lambda_reg * self.weights_hidden1_hidden2)
        self.m_hidden2_output, self.v_hidden2_output, dw_hidden2_output = adam_optimizer(self.m_hidden2_output, self.v_hidden2_output, t, self.hidden_layer2_output.T.dot(output_delta) + self.lambda_reg * self.weights_hidden2_output)

        self.m_bias_hidden1, self.v_bias_hidden1, db_bias_hidden1 = adam_optimizer(self.m_bias_hidden1, self.v_bias_hidden1, t, np.sum(hidden1_delta, axis=0, keepdims=True))
        self.m_bias_hidden2, self.v_bias_hidden2, db_bias_hidden2 = adam_optimizer(self.m_bias_hidden2, self.v_bias_hidden2, t, np.sum(hidden2_delta, axis=0, keepdims=True))
        self.m_bias_output, self.v_bias_output, db_bias_output = adam_optimizer(self.m_bias_output, self.v_bias_output, t, np.sum(output_delta, axis=0, keepdims=True))

        # Update weights and biases
        self.weights_input_hidden1 -= dw_input_hidden1
        self.weights_hidden1_hidden2 -= dw_hidden1_hidden2
        self.weights_hidden2_output -= dw_hidden2_output

        self.bias_hidden1 -= db_bias_hidden1
        self.bias_hidden2 -= db_bias_hidden2
        self.bias_output -= db_bias_output

        return regularization_term

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            t = epoch + 1
            output = self.forward(X)
            regularization_term = self.backward(X, y, learning_rate, t)

            loss = np.mean(np.square(y - output)) + regularization_term
            self.losses.append(loss)

            if loss < self.best_loss:
                self.best_loss = loss
                self.early_stop_count = 0
            else:
                self.early_stop_count += 1

            if self.early_stop_count >= self.patience:
                print(f"Early stopping at epoch {epoch}, loss: {loss}")
                break

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

nn = NeuralNetwork(input_size=X_train_scaled.shape[1], hidden_size1=20, hidden_size2=10, output_size=1, lambda_reg=0.01, patience=50)
nn.train(X_train_scaled, y_train.values.reshape(-1, 1), epochs=10000, learning_rate=0.001)

# Test the Neural Network
y_pred_prob = nn.forward(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Plot Loss over Epochs
plt.figure(figsize=(8, 6))
plt.plot(nn.losses)
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Visualize confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Actual vs Predicted Class Distribution
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.countplot(x=y_test, palette='Blues')
plt.title("Actual Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
sns.countplot(x=y_pred.flatten(), palette='Greens')
plt.title("Predicted Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

