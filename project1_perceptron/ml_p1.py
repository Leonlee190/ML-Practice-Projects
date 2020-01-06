# Project: CS 445 Project 1
# Programmer: SeungJun Lee (Leon)
# Description: Training and testing perceptrons to recognize one handwritten number between 0 to 9

import pandas as pd
import numpy as np
import random
import time
import csv

# Learning rate
learn = 1

# Accuracy counter and array to hold all accuracy of epochs
correct = 0
accuracy_array = np.zeros((2, 51))

# Output y value holder
output_result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Dot product value holder
output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Confusion matrix
confusion = np.zeros((10, 10))

# Importing csv data as numpy array
train_data = np.array(pd.read_csv('mnist_train.csv', header=None), dtype='float')
test_data = np.array(pd.read_csv('mnist_test.csv', header=None), dtype='float')

# Trimming training and testing data's grayscale value to be between 0 and 1
for i in range(len(train_data)):
    train_data[i, 1:] = train_data[i, 1:] / 255

for i in range(len(test_data)):
    test_data[i, 1:] = test_data[i, 1:] / 255

# Setting up bias input value by inserting before the grayscale values
train_data = np.insert(train_data, 1, 1, axis=1)
test_data = np.insert(test_data, 1, 1, axis=1)

# Setting up weight array for 10 perceptrons and 785 inputs per perceptron
weights = np.array([[0] * 785 for i in range(10)], dtype='float')

# Setting random weight values between -0.5 and 0.5 for each input
for x in np.nditer(weights, op_flags=['readwrite']):
    rand = round(random.uniform(-0.5, 0.5), 1)

    # If the random value is 0 then just set it as 0.2
    if rand == 0:
        rand = 0.2
    x[...] = rand + x

# Setting time to check how long it took to train & test
start = time.time()

# 51 Epoch which includes 0th Epoch
for epoch in range(51):
    # Reset correct fraction
    correct = 0

    # Loop through each training data
    for i in range(len(train_data)):
        # Set up target value and set the current training data's correct value as 1
        target = np.zeros(len(weights), dtype=float)
        target[int(train_data[i, 0])] = 1

        # For one training data, loop through all 0 to 9 perceptrons
        for k in range(len(weights)):
            # Dot product between one perceptron and one training data's grayscale
            output[k] = np.dot(train_data[i, 1:], weights[k])

            # If the dot product is bigger than 0 then y is 1 else 0
            if output[k] > 0:
                output_result[k] = 1
            else:
                output_result[k] = 0

        # If the training data's correct value is same as index of maximum value of dot products
        # then it's correct so increment correct
        if train_data[i, 0] == output.index(max(output)):
            correct += 1

        # Ignore 0th Epoch and start updating weight from 1st Epoch
        if epoch > 0:
            # Go through target array
            for h in range(len(target)):
                # subtract perceptron's y value from target's value
                diff = target[h] - output_result[h]

                # If there is difference between the two
                if diff != 0:
                    # Then go through all the perceptron's weight values
                    for w in range(len(weights[h])):
                        # wi = wi + n*(t^k - y^k) * xi^k
                        update = learn * diff * train_data[i, w+1]
                        weights[h, w] = weights[h, w] + update

    # Get this epoch's training set's accuracy and put it in an array
    train_acc = correct / (len(train_data)) * 100
    accuracy_array[0, epoch] = train_acc
    print("Training epoch ", epoch, " accuracy = ", train_acc)

    # Reset accuracy counter for testing epoch
    correct = 0

    # Loop through all the test data values
    for test_range in range(len(test_data)):
        # Set up target array and initialize it with test data's correct value
        target = np.zeros(len(weights), dtype=float)
        target[int(test_data[test_range, 0])] = 1

        # Go through all the perceptrons
        for test_weight in range(len(weights)):
            # Dot product between test data and each perceptron's grayscale
            output[test_weight] = np.dot(test_data[test_range, 1:], weights[test_weight])

            # if the dot product is higher than zero then y is 1 else 0
            if output[test_weight] > 0:
                output_result[test_weight] = 1
            else:
                output_result[test_weight] = 0

        # Increment confusion matrix's index
        # test data's correct value as row
        # output's prediction as column
        confusion[int(test_data[test_range, 0]), int(output.index(max(output)))] += 1

        # if the correct data matches prediction then increment
        if test_data[test_range, 0] == output.index(max(output)):
            correct += 1

    # Calculate and store this epoch's test accuracy
    test_acc = correct / (len(test_data)) * 100
    accuracy_array[1, epoch] = test_acc
    print("Testing epoch ", epoch, " accuracy = ", test_acc)
    print("----------------------------------------------------------")

# Printing how many min it took to do the 51 epoch
print("Time: ", (time.time() - start)/60)

# Writing the training & test accuracy array into csv file
graphFile = open('accuracy_graph.csv', 'w')
with graphFile:
    writer = csv.writer(graphFile)
    writer.writerows(accuracy_array)

# Confusion matrix into csv file
confusionFile = open('confusion_matrix.csv', 'w')
with confusionFile:
    writer = csv.writer(confusionFile)
    writer.writerows(confusion)
