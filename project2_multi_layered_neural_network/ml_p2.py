# Project: CS 445 Project 2
# Programmer: SeungJun Lee (Leon)
# Description: Training and testing multi-layered neural network to recognize one handwritten number between 0 to 9

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Importing csv data as numpy array
train_data = np.array(pd.read_csv('mnist_train.csv', header=None), dtype='float')
test_data = np.array(pd.read_csv('mnist_test.csv', header=None), dtype='float')

# Constants for Machine Learning Settings
learning_rate = 0.1
momentum = 0.9
training_amount = int(len(train_data)/2)

num_input = 784
num_hidden = 100
num_output = 10

# Accuracy calculation
correct_accuracy = 0
train_accuracy = np.zeros(51)
test_accuracy = np.zeros(51)
axis = np.arange(1, 52)
confusion = np.zeros((10, 10))

# Delta values
hidden_to_output_deltas = np.zeros((num_output, num_hidden + 1), dtype=float)
input_to_hidden_deltas = np.zeros((num_hidden, num_input + 1), dtype=float)


def sigmoid(x):
    if x < 0:
        return float(1 - 1 / (1 + math.exp(x)))
    return float(1 / (1 + math.exp(-x)))


# Trimming training and testing data's gray scale value to be between 0 and 1
for i in range(len(train_data)):
    train_data[i, 1:] = train_data[i, 1:] / 255

for i in range(len(test_data)):
    test_data[i, 1:] = test_data[i, 1:] / 255

# Setting up bias input value by inserting before the gray scale values
train_data = np.insert(train_data, 1, 1, axis=1)
test_data = np.insert(test_data, 1, 1, axis=1)

# Setting up input to hidden weight array and rounding it to 2nd decimal
input_to_hidden_weights = np.random.uniform(low=-0.5, high=0.5, size=(num_hidden, (num_input + 1)))
input_to_hidden_weights = np.around(input_to_hidden_weights, decimals=2)

# Setting up hidden to output weight array and rounding it to 2nd decimal
hidden_to_output_weights = np.random.uniform(low=-0.5, high=0.5, size=(num_output, (num_hidden + 1)))
hidden_to_output_weights = np.around(hidden_to_output_weights, decimals=2)

for epoch in range(51):
    correct_accuracy = 0

    # Start of the training
    for iter_train in range(training_amount):
        targets = np.full(10, 0.1, dtype=float)
        targets[int(train_data[iter_train][0])] = 0.9

        # Hidden activation calculated through applying sigmoid function to
        # individual dot product between input to hidden weights and inputs
        input_hidden_dot = input_to_hidden_weights.dot(train_data[iter_train, 1:])
        hidden_activation = np.fromiter(map(sigmoid, input_hidden_dot), dtype=float)
        hidden_activation = np.insert(hidden_activation, 0, 1)

        # Output activation calculated by applying sigmoid function to
        # individual dot product between hidden to output weights and the hidden activation values
        hidden_output_dot = hidden_to_output_weights.dot(hidden_activation)
        output_activation = np.fromiter(map(sigmoid, hidden_output_dot), dtype=float)

        # Check if target and guess is same
        correct = train_data[iter_train, 0]
        guess = np.argmax(output_activation)
        if correct == guess:
            correct_accuracy += 1

        # Calculating error terms for output
        output_error = np.fromiter(map(lambda x, y: x*(1-x)*(y-x), output_activation, targets), dtype=float)

        # Calculating error terms for hidden unit after summing
        sum_hidden = np.array(np.sum((hidden_to_output_weights[:, 1:]*output_error[:, None]), axis=0))
        hidden_error = np.fromiter(map(lambda x, y: x*(1-x)*y, hidden_activation[1:], sum_hidden), dtype=float)

        # Updating hidden to output delta value
        hidden_to_output_deltas = hidden_to_output_deltas * momentum
        hidden_to_output_deltas = learning_rate * np.outer(output_error, hidden_activation) + hidden_to_output_deltas

        # Updating hidden to output weight with delta
        hidden_to_output_weights = hidden_to_output_weights + hidden_to_output_deltas

        # Updating input to hidden delta value
        input_to_hidden_deltas = input_to_hidden_deltas * momentum
        input_delta_set = np.outer(hidden_error, train_data[iter_train, 1:])
        input_to_hidden_deltas = (learning_rate * input_delta_set) + input_to_hidden_deltas

        # Updating input to hidden weight with delta
        input_to_hidden_weights = input_to_hidden_weights + input_to_hidden_deltas

    # Accuracy calculation
    train_acc = (correct_accuracy / training_amount) * 100
    train_accuracy[epoch] = train_acc
    print("Training epoch ", epoch, " accuracy = ", round(train_acc, 2))

    correct_accuracy = 0
    for iter_test in range(len(test_data)):
        targets = np.full(10, 0.1, dtype=float)
        targets[int(test_data[iter_test][0])] = 0.9

        # Hidden activation calculated through applying sigmoid function to
        # individual dot product between input to hidden weights and inputs
        hidden_activation = np.fromiter(map(sigmoid, input_to_hidden_weights.dot(test_data[iter_test][1:])), dtype=float)
        hidden_activation = np.insert(hidden_activation, 0, 1)

        # Output activation calculated by applying sigmoid function to
        # individual dot product between hidden to output weights and the hidden activation values
        hidden_output_dot = hidden_to_output_weights.dot(hidden_activation)
        output_activation = np.fromiter(map(sigmoid, hidden_output_dot), dtype=float)

        # Check if target and guess is same
        correct = test_data[iter_test, 0]
        guess = np.argmax(output_activation)
        if correct == guess:
            correct_accuracy += 1

        confusion[int(test_data[iter_test, 0]), np.argmax(output_activation)] += 1

    # Accuracy calculation
    test_acc = (correct_accuracy / len(test_data)) * 100
    test_accuracy[epoch] = test_acc
    print("Testing epoch ", epoch, " accuracy = ", round(test_acc, 2))
    print("------------------------------------------------------------------------")

# Plotting Training and Test result accuracy per epoch
plt.plot(axis, train_accuracy, label="Training")
plt.plot(axis, test_accuracy, label="Test")

# Setting the plot and saving the graph
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title('Learning rate = ' + str(learning_rate) + " , Momentum = " + str(momentum) + ' , Hidden Unit = ' + str(num_hidden) + ' , Training amount = ' + str(training_amount))
plt.legend()
plt.savefig('lr=' + str(learning_rate) + ",mom=" + str(momentum) + ',Hidden Unit=' + str(num_hidden) + ',tr=' + str(training_amount) + '.png', bbox_inches='tight')

# Plotting the confusion matrix with heat map and saving the table
plt.figure(figsize=(20, 5))
plt.ticklabel_format(useOffset=False, style='plain')
df = pd.DataFrame(data=confusion, index=np.arange(10), columns=np.arange(10))
plt.title('Learning rate = ' + str(learning_rate) + " , Momentum = " + str(momentum) + ' , Hidden Unit = ' + str(num_hidden) + ' , Training amount = ' + str(training_amount))
svm = sns.heatmap(df, annot=True, linewidths=0.5, cmap="Blues", fmt='.1f')
plt.savefig('Confusion,lr=' + str(learning_rate) + ",mom=" + str(momentum) + ',Hidden Unit=' + str(num_hidden) + ',tr=' + str(training_amount) + '.png')
