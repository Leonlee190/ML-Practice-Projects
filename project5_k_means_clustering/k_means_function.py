import numpy as np
import math


# Function file for k-Mean cluster



# While statement ender
def check_equal(x, y):
    return np.array_equal(x, y)


# returns the non square rooted euclidean value
def euclidean_double(x, m):
    return np.sum(np.power(np.subtract(x, m), 2))


# returns euclidean value of two points
def euclidean(x, m):
    return math.sqrt(np.sum(np.power(np.subtract(x, m), 2)))


# apply euclidean with one data point and all the cluster points to see which fits best
def class_selection(x, m, k):
    selection = np.zeros(k)

    for i in range(k):
        selection[i] = euclidean(x, m[i])

    return np.argmin(selection)


# apply the euclidean to all the data points
def all_class_selection(x, m, k):
    selection = np.zeros(len(x))

    for i in range(len(x)):
        selection[i] = class_selection(x[i], m, k)

    return selection


# returns the updated cluster centers
def calc_changes(x, m, chosen, k):
    count = np.zeros(k)
    summing = np.zeros((k, 64))

    # get all the points in the cluster and add up
    for i in range(len(chosen)):
        chosen_index = chosen[i]
        summing[int(chosen_index), :] += x[int(i), :]
        count[int(chosen_index)] += 1

    for i in range(k):
        if count[i] == 0:
            count[i] = 1

    # average and return
    return np.array(summing / count[:, None])


# retrieve the mapping of each cluster's class values into confusion matrix
def get_cluster_confusion(chosen, k, target):
    cluster_matrix = np.zeros((k, 10))

    for i in range(len(chosen)):
        x = chosen[i]
        y = target[i]
        cluster_matrix[int(x), int(y)] += 1

    return cluster_matrix


# decide which class will each cluster will be set as
def set_cluster_class(chosen, k, target):
    cluster_matrix = get_cluster_confusion(chosen, k, target)
    cluster_class = np.zeros(k)

    # take the max occurrence class and set it
    for i in range(k):
        cluster_class[i] = np.argmax(cluster_matrix[i])

    return cluster_class


# calculate the mean square error value
def mse(x, m, chosen, k):
    count = np.zeros(k)
    mse_matrix = np.zeros(k)

    # get the non-square rooted value of all the points in the cluster with the cluster center
    for i in range(len(chosen)):
        this_time = chosen[i]

        count[int(this_time)] += 1
        mse_matrix[int(this_time)] = euclidean_double(x[i], m[int(this_time)])

    # just in case there is 0 and to avoid division by 0 set it as 1
    for i in range(len(mse_matrix)):
        if count[i] == 0:
            count[i] = 1

        if mse_matrix[i] == 0:
            mse_matrix[i] = 1

    mse_final = np.array(mse_matrix / count)

    return mse_final


# get the average mean square error
def avg_mse(mse_matrix):
    averaging = np.sum(mse_matrix)
    averaging = averaging / len(mse_matrix)

    return averaging


# get the mean square separation
def mss(m, k):
    total = 0

    # loop through all and calculate euclidean value of all each other
    for i in range(len(m)):
        for j in range(len(m)):
            if i != j:
                total += euclidean_double(m[i], m[j])

    deno = k * (k - 1) / 2

    return total / deno


# calculating one cluster's entropy value
def individual_entropy(cluster):
    count = np.zeros(10)

    # getting the number of repeated occurrences
    for i in range(10):
        count[i] = (cluster == i).sum()

    total = np.sum(count)
    adding = 0

    # calculate the possibility times log2 of the possibility
    for i in range(10):
        cur = count[i]
        if cur != 0:
            adding = adding + (cur/total) * math.log((cur/total), 2)

    # return negative value of it
    return 0 - adding


# get entropy value of all clusters
def entropy(c_map):
    size = len(c_map)
    entropy_matrix = np.zeros(size)

    # call entropy for all clusters
    for i in range(size):
        entropy_matrix[i] = individual_entropy(c_map[i])

    return entropy_matrix


# get mean entropy value
def mean_entropy(cluster, entropy_value):
    add = 0
    total = np.sum(cluster)

    # sum up the value of entropy * possibility
    for i in range(len(entropy_value)):
        # Getting the possibility
        current_sum = np.sum(cluster[i])
        current_sum = current_sum / total

        # multiply by the entropy
        add = add + current_sum * entropy_value[i]

    return add



