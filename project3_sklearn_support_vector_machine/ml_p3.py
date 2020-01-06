from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import precision_recall_curve

# Reading data from the csv
data_file_name = "spambase.csv"
spam_data_set = np.array(pd.read_csv(data_file_name, header=None), dtype=float)

# Shuffling randomly
np.random.shuffle(spam_data_set)

# Mid point separator
middle_point = int(len(spam_data_set)/2)

# Half it for training and testing
train_set = np.array(spam_data_set[0:middle_point, :])
test_set = np.array(spam_data_set[middle_point:, :])

# Get the scaler value which has the mean and variance from the training set
scaler = preprocessing.StandardScaler().fit(train_set[:, :57])

# Scale the training and separate the training target
train_scaled_set = np.array(scaler.transform(train_set[:, :57]))
train_target = np.array(train_set[:, 57])

# Scale the test set with training data's scale and separate the testing target
test_data = np.array(test_set[:, :57])
test_scaled_set = np.array(scaler.transform(test_data))
test_target = np.array(test_set[:, 57])

# Train Linear Kernel and get the result of the test
clf = svm.SVC(kernel='linear', gamma='scale')
clf.fit(train_scaled_set, train_target)
result = clf.predict(test_scaled_set)

# Retrieve the weight values
weight = clf._get_coef()

# Retrieve score
accuracy = clf.score(test_scaled_set, test_target)
precision, recall, threshold = precision_recall_curve(test_target, result)

print("Experiment 1 Accuracy: " + str(accuracy*100))
print("Precision: " + str(precision))
print("Recall: " + str(recall))

# Get False Positive and True Positive to plot ROC curve
fpr, tpr, threshold = metrics.roc_curve(test_target, result)

plt.figure()
plt.plot(fpr, tpr, color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Experiment 1: ROC')
plt.savefig('Experiment1.png')

# Set up accuracy array for plot
accuracy_m = np.zeros(57-2)

# Experiment 2
for m in range(2, 57):
    # Make weight absolute value then pick out the top mth biggest
    new_weight = np.array(np.absolute(weight))
    feature_index = np.argpartition(new_weight, m)
    select_feature = np.array(feature_index[0, :m])

    # Make new training and testing data set with m amount of features
    new_train_set = np.copy(train_scaled_set[:, select_feature])
    new_test_set = np.copy(test_scaled_set[:, select_feature])

    # Train then predict with the new training and testing data set
    clf = svm.SVC(kernel='linear')
    clf.fit(new_train_set, train_target)
    result_feature = clf.predict(new_test_set)
    accuracy_feature = clf.score(new_test_set, test_target)
    # precision_feature, recall_feature, threshold_feature = precision_recall_curve(test_target, result_feature)
    accuracy_m[m-2] = accuracy_feature*100

# Create and save Experiment 2 plot
plt.figure()
plt.plot(range(1, 56), accuracy_m)
plt.xlabel('Feature m')
plt.ylabel('Accuracy')
plt.title('Experiment 2: Accuracy v. Feature m')
plt.savefig('Experiment2.png')

# Experiment 3
for m in range(2, 57):
    # Create feature indices then shuffle them
    feature_index = np.array(range(57))
    np.random.shuffle(feature_index)
    select_feature = np.array(feature_index[:m])

    # Pick out m amount of feature
    new_train_set = np.copy(train_scaled_set[:, select_feature])
    new_test_set = np.copy(test_scaled_set[:, select_feature])

    # Train and test with the new data
    clf = svm.SVC(kernel='linear')
    clf.fit(new_train_set, train_target)
    result_feature = clf.predict(new_test_set)
    accuracy_feature = clf.score(new_test_set, test_target)
    # precision_feature, recall_feature, threshold_feature = precision_recall_curve(test_target, result_feature)
    accuracy_m[m-2] = accuracy_feature*100

# Plot and save Experiment 3
plt.figure()
plt.plot(range(1, 56), accuracy_m)
plt.xlabel('Feature m')
plt.ylabel('Accuracy')
plt.title('Experiment 3: Accuracy v. Feature m')
plt.savefig('Experiment3.png')
