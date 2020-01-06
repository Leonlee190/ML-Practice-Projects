import numpy as np
import pandas as pd
import math

# mean calculation
def mean_calc(arr, class_value):
    # selecting the class valued 0 or 1 data
    mean_arr = np.array(arr[np.where(arr[:, 57] == class_value)])
    mean_arr = np.array(mean_arr[:, :57])
    mean_size = len(mean_arr)

    # summing then finding mean
    calc_mean = np.array(mean_arr.sum(axis=0))
    calc_mean = calc_mean/mean_size

    return calc_mean


# standard deviation calculation
def stand_devi_calc(arr, mean_values, class_value):
    # selecting the class valued 0 or 1 data
    stand_arr = np.array(arr[np.where(arr[:, 57] == class_value)])
    stand_arr = np.array(stand_arr[:, :57])
    stand_size = len(stand_arr)

    # subtract mean value of that feature then calculating
    calc_stand = stand_arr - mean_values[None, :]
    calc_stand = np.array(np.power(calc_stand, 2))
    calc_stand = np.array(calc_stand.sum(axis=0))
    calc_stand = calc_stand/stand_size
    calc_stand = np.array(np.sqrt(calc_stand))

    # adding epsilon
    calc_stand = np.array(calc_stand + 0.0001)

    return calc_stand


# gaussian naive bayes algorithm
def gauss_calc(x, mean_value, stand_value):
    first = 1 / (math.sqrt(2*math.pi)*stand_value)
    second = 0 - (math.pow((x - mean_value), 2) / (2 * math.pow(stand_value, 2)))
    third = math.exp(second)
    final = first * third

    # if the final result is 0 then spit out large negative number so it won't do log(0)
    if final == 0:
        return -999999
    else:
        return math.log(final)


# Reading data from the csv
data_file_name = "spambase.csv"
spam_data_set = np.array(pd.read_csv(data_file_name, header=None), dtype=float)

# Randomizing and splitting into test and train and target
np.random.shuffle(spam_data_set)
middle_point = int(len(spam_data_set)/2)
train_set = np.array(spam_data_set[0:middle_point, :])
test_set = np.array(spam_data_set[middle_point:, :])
train_data = np.array(train_set[:, :57])
test_data = np.array(test_set[:, :57])
train_target = np.array(train_set[:, 57])
test_target = np.array(test_set[:, 57])

# Getting the class probability
train_pos = 0
for i in range(len(train_target)):
    if train_target[i] == 1:
        train_pos += 1

train_pos = train_pos / len(train_target)
train_neg = 1 - train_pos

# retrieving mean value of each feature for both positive and negative
train_mean_pos = mean_calc(train_set, 1)
train_mean_neg = mean_calc(train_set, 0)

# retrieving standard deviation value of each feature for both positive and negative
train_stand_pos = stand_devi_calc(train_set, train_mean_pos, 1)
train_stand_neg = stand_devi_calc(train_set, train_mean_neg, 0)

# result after argmax
test_result = np.zeros(len(test_set))

# go through all the test set
for i in range(len(test_set)):
    # calculate all the feature possibilities
    test_pos = map(gauss_calc, test_data[i, :], train_mean_pos, train_stand_pos)
    test_neg = map(gauss_calc, test_data[i, :], train_mean_neg, train_stand_neg)

    pos_arr = np.fromiter(test_pos, dtype=float)
    neg_arr = np.fromiter(test_neg, dtype=float)

    # get the sum of all possibilities
    pos = math.log(train_pos) + pos_arr.sum(dtype=float)
    neg = math.log(train_neg) + neg_arr.sum(dtype=float)

    # if positive is bigger than 1 else 0
    if pos > neg:
        test_result[i] = 1
    else:
        test_result[i] = 0

# confusion matrix
confusion = np.zeros((2, 2))

# Getting the accuracy
correct = 0
for i in range(len(test_target)):
    actual = int(test_target[i])
    predict = int(test_result[i])
    confusion[actual, predict] += 1
    if test_result[i] == test_target[i]:
        correct += 1

recall = confusion[1, 1] / (confusion[1, 1] + confusion[1, 0])
precision = confusion[1, 1] / (confusion[1, 1] + confusion[0, 1])
accuracy = correct/len(test_target) * 100

# Print out accuracy, recall, precision, and confusion matrix
print("Accuracy: ", accuracy)
print("Recall: ", recall)
print("Precision: ", precision)
print("Confusion Matrix: ")
print(confusion)
