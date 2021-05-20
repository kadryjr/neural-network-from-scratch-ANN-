# Loai Gamal Mohamed        20180206    G1
# Ahmed Kadry Abd El-Shafy  20180018    G1
# Assignment2 Supervised Learning (Neural Network)


import numpy as np
import matplotlib.pyplot as plt


class Layer:
    def __init__(self, number_of_inputs, number_of_neurons):
        self.weights = 0.1 * np.random.randn(number_of_inputs, number_of_neurons)  # initialize random weights
        self.biases = np.zeros((1, number_of_neurons))  # initialize biases with 0s

    def feed_forward(self, inputs_array):
        self.output = np.dot(inputs_array, self.weights) + self.biases


# Output of j as in lecture
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# output j * (1 - output j)
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def unit_step(x):
    output = x
    rows, columns = x.shape
    for i in range(rows):
        for j in range(columns):
            if output[i][j] < 0:
                output[i][j] = 0
            else:
                output[i][j] = 1
    return output


# non-Linear Points
# X = np.array([[1, 1], [2.3, 1], [2, 2], [4, 2], [0, 3], [3, 3], [2, 4], [3.8, 3.3], [1.5, 2.5], [0.5, 2], [0, 1.5]
#                  , [0.5, 2.5], [3.5, 2.5], [0, 2.5], [1, 3.5], [1.5, 1.5], [2.5, 3.5], [3.3, 1.5], [2.8, 2.5],
#               [3.3, 3.8], [2.3, 2.8], [1.8, 3],
#               [1, 2.2], [1.3, 3]])

# target = np.array([[0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]])

# Linearly Separable Points
X = np.array([[1, 1], [3, 1], [2, 2], [4, 2], [0, 3], [3, 3], [2, 4], [4, 4], [1.5, 2.5], [0.5, 2], [0, 1.5],
              [0.5, 2.5], [3.5, 2.5], [0, 2.5], [1, 3.5], [1.5, 1.5], [2.5, 3.5], [3.3, 1.5], [2.8, 2.5],
              [3.3, 3.8]])

target = np.array([[0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]])  # 1 row & 20  columns

rows, columns = target.shape  # target[1][20]
target = target.reshape(columns, rows)  # target[20][1] (rotate rows with columns)

# constant (ETA)
learning_rate = 0.005

# initializing number of neurons in a layer and inputs
n_neurons, n_inputs = X.shape


layer1 = Layer(n_inputs, n_neurons)     # Hidden Layer
layer2 = Layer(n_neurons, n_neurons)    # output layer

least_error = 10
l_error_index = 0
optimal_W = layer1.weights
optimal_biases = layer1.biases

for epoch in range(100):
    layer1.feed_forward(X)
    # feedforward input
    l1_ff_input = layer1.output

    # Layer 1 feedforward output
    l1_ff_output = sigmoid(l1_ff_input)

    # Layer 2
    layer2.feed_forward(l1_ff_output)

    l2_ff_input = layer2.output

    l2_ff_output = unit_step(l2_ff_input)
    print(l2_ff_output)

    # BACKPROPAGATION
    error = target - l2_ff_output   # error = t j - o j)
    sumed_error = 0.5 * (error.sum()) ** 2
    print("error ", sumed_error)
    if sumed_error < least_error:
        least_error = sumed_error
        l_error_index = epoch
        optimal_W = layer1.weights
        optimal_biases = layer1.biases
    # calculating derivatives
    template_error = error  # = ( t j - o j )
    deriv_of_error = sigmoid_derivative(l2_ff_output)  # o j (1 - o j)

    # △w= learning rate * delta * prevLayer output = eta * o j (1 - o j) o i
    delta = template_error * deriv_of_error

    # transpose of X
    inputs = X.transpose()

    # updating weights
    layer1.weights -= learning_rate * np.dot(inputs, delta)
    # updating biases
    for j in delta:
        layer1.biases -= learning_rate * j

# print the optimal weights and biases at the least error
print(layer1.weights)
print(optimal_W, "\n")
print(layer1.biases)
print(layer1.biases, "\n")

print("Least Error:", least_error)
print("Least Error loop number:", l_error_index)


test_point = np.array([2, 3])
result = np.dot(test_point, optimal_W) + optimal_biases
result = sigmoid(result)
final_result = result.mean()

if final_result < 0.5:
    print("Result = ", final_result)
    print("Classified to class 0")
else:
    print("Result = ", final_result)
    print("Classified to class 1")

plt.scatter(X[:, 0], X[:, 1], c=target)
plt.show()
