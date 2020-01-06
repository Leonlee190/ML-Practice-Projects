import numpy as np
import pandas as pd
import k_means_function as km
import seaborn as sns
import matplotlib.pyplot as plt


# Project: CS 445 HW 5
# Programmer: SeungJun Lee
# Description: K-Mean Cluster learning
# When compiled and executed it runs two cluster number of 10 and 30
# Each 10 and 30 cluster number will run 5 times each
# Each iteration will save the heatmap of the cluster and confusion matrix
# And it will print out various metric and accuracy



k = 0

# Read training and testing data
train = np.array(pd.read_csv("optdigits.train", header=None), dtype=float)
test = np.array(pd.read_csv("optdigits.test", header=None), dtype=float)

# separate them into data and target
train_data = np.array(train[:, :64])
test_data = np.array(test[:, :64])
train_target = np.array(train[:, 64])
test_target = np.array(test[:, 64])

# need to do two sets of five trials
for total in range(2):
    # first one is k value 10 and second time is with 30
    if total == 0:
        k = 10
    else:
        k = 30

    # gotta repeat it five times per k value
    for epoch in range(5):
        # create random starting cluster points
        clustering = np.random.randint(17, size=(k, 64))
        chosen = np.zeros(len(train_data))

        # initial start of finding the center of cluster
        chosen = km.all_class_selection(train_data, clustering, k)
        old_m = km.calc_changes(train_data, clustering, chosen, k)
        # put in old cluster m value into the euclidean to get updated ones
        chosen = km.all_class_selection(train_data, old_m, k)
        new_m = km.calc_changes(train_data, old_m, chosen, k)

        # do center finding till the old is same as new
        while not km.check_equal(old_m, new_m):
            # old will be updated with new iteration
            old_m = new_m
            chosen = km.all_class_selection(train_data, old_m, k)
            new_m = km.calc_changes(train_data, old_m, chosen, k)

        cluster_m = np.array(new_m)

        # get which cluster means what class
        cluster_class = km.set_cluster_class(chosen, k, train_target)

        # calculate mse, mse average, mss, entropy, and mean entropy
        mse = km.mse(train_data, cluster_m, chosen, k)
        mse_avg = km.avg_mse(mse)
        mss = km.mss(cluster_m, k)
        cluster_confusion = km.get_cluster_confusion(chosen, k, train_target)
        entropy_matrix = km.entropy(cluster_confusion)
        mean_entropy = km.mean_entropy(cluster_confusion, entropy_matrix)

        # print the calculated values
        print("\n----------------------------------------------------------------")
        print("K = ", k)
        print("Epoch: ", epoch+1)
        print("MSE:")
        with np.printoptions(precision=3, suppress=True):
            print(mse)

        print("\nAVG MSE:", mse_avg)
        print("\nMSS:", mss)
        print("\nEntropy:")
        with np.printoptions(precision=3, suppress=True):
            print(entropy_matrix)

        print("\nMean Entropy:", mean_entropy)

        # retrieve what the cluster results of test data was
        test_chosen = km.all_class_selection(test_data, cluster_m, k)

        confusion_matrix = np.zeros((10, 10))
        accuracy = 0

        # create confusion matrix and calculate accuracy with the result given from test class comparison
        for i in range(len(test_chosen)):
            index = test_chosen[i]
            test_result = cluster_class[int(index)]
            true_result = test_target[i]

            confusion_matrix[int(true_result), int(test_result)] += 1

            if test_result == true_result:
                accuracy += 1

        print("\nAccuracy:", accuracy / len(test_target))

        # visualization of the cluster centers via heatmap
        for i in range(len(cluster_m)):
            cluster_section = np.array(cluster_m[i])
            cluster_section = cluster_section.astype(int)
            cluster_section = cluster_section.reshape(8, 8)
            df = pd.DataFrame(data=cluster_section, index=np.arange(8), columns=np.arange(8))
            svm = sns.heatmap(df, annot=True, linewidths=0.5, cmap="Blues", fmt='.0f', cbar=False)
            plt.savefig('cluster/k' + str(k) + '/Cluster Visualization. K = ' + str(k) + ', Epoch = ' + str(epoch+1) + ', Cluster = ' + str(i) + '.png')
            plt.clf()

        # draw out the confusion matrix
        plt.figure(figsize=(10, 10))
        df = pd.DataFrame(data=confusion_matrix, index=np.arange(10), columns=np.arange(10))
        svm = sns.heatmap(df, annot=True, linewidths=0.5, cmap="Blues", fmt='.0f', cbar=False)
        plt.savefig('confusion/k' + str(k) + '/Confusion Matrix. K = ' + str(k) + ', Epoch = ' + str(epoch+1) + '.png')
        plt.clf()
