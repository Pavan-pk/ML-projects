#!/usr/bin/env python
# encoding: utf-8

__author__ = "Pavan Kumar Raja"
__version__ = "1.0"

import math
import numpy as np
import scipy.io
import statistics as st
from collections import defaultdict
from tqdm import tqdm


class NaiveBayesClassifier:
    """
    Naive bayes classifier.
    """

    def __init__(self, train_x, train_y, test_x, test_y):
        """
        :param train_x: train data features
        :param train_y: test data labels
        :param test_x: train data features
        :param text_y: test data labels
        """
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def calculate_prob_x_in_dist(self, x, mean, std):
        """
        Calculate probability of x belonging to normal distribution (Gaussian) function with (mean, std)
        :param x: feature value
        :param mean: mean of the normal distribution
        :param std: std of the normal distribution
        :return: probability calculated
        """
        height = 1 / (std * math.sqrt(2 * math.pi))
        distribution = math.exp(-0.5 * math.pow((x - mean) / std, 2))
        return height * distribution

    def naive_bayes_classifier(self):
        """
        Create a Naive bayes classifier and calculate accuracy of the classifier for the given dataset.
        :return: (int, int) -> Number of correct predictions and wrong predictions on test dataset.
        """
        # Shape of train_y = (1, 12116)
        train_labels = [int(x) for x in self.train_y[0]]
        train_separated_by_class = defaultdict(list)
        for idx, image in enumerate(self.train_x):
            train_separated_by_class[train_labels[idx]].append([np.mean(image), np.std(image)])

        # Calculate priors p(y)
        zero_labels = len(train_separated_by_class[0])  # count 7s
        one_labels = len(train_separated_by_class[1])  # count 8s
        prior_zero = zero_labels / (zero_labels + one_labels)
        prior_one = one_labels / (zero_labels + one_labels)

        # Calculate p(xi=vk/y)
        # We have 2 features i.e means and standard deviation.
        # Need to have 2 distribution for each labels to calculate posterior probability.

        zero_mean_std = list(zip(*train_separated_by_class[0]))
        one_mean_std = list(zip(*train_separated_by_class[1]))

        zero_dist_mean_mean = sum(zero_mean_std[0]) / len(zero_mean_std[0])
        zero_dist_mean_std = st.stdev(zero_mean_std[0])
        zero_dist_std_mean = sum(zero_mean_std[1]) / len(zero_mean_std[1])
        zero_dist_std_std = st.stdev(zero_mean_std[1])

        one_dist_mean_mean = sum(one_mean_std[0]) / len(one_mean_std[0])
        one_dist_mean_std = st.stdev(one_mean_std[0])
        one_dist_std_mean = sum(one_mean_std[1]) / len(one_mean_std[1])
        one_dist_std_std = st.stdev(one_mean_std[1])

        test_y_labels = [int(x) for x in self.test_y[0]]
        predictions = []
        correct = defaultdict(int)

        for idx, image in enumerate(self.test_x):
            mean = np.mean(image)
            std = np.std(image)

            zero_prob_mean = self.calculate_prob_x_in_dist(mean, zero_dist_mean_mean, zero_dist_mean_std)
            zero_prob_std = self.calculate_prob_x_in_dist(std, zero_dist_std_mean, zero_dist_std_std)

            one_prob_mean = self.calculate_prob_x_in_dist(mean, one_dist_mean_mean, one_dist_mean_std)
            one_prob_std = self.calculate_prob_x_in_dist(std, one_dist_std_mean, one_dist_std_std)

            predict_seven = prior_zero * zero_prob_mean * zero_prob_std
            predict_eight = prior_one * one_prob_mean * one_prob_std

            if predict_seven > predict_eight:
                predict = 0
            else:
                predict = 1
            predictions.append(predict)
            if predict == test_y_labels[idx]:
                correct['total'] += 1
                correct[predict] += 1

        print(prior_zero, prior_one)
        return correct['total'] / len(predictions), \
               correct[0] / sum(x == 0 for x in test_y_labels), \
               correct[1] / sum(x == 1 for x in test_y_labels)

    def get_naive_bayes_accuracy(self):
        accuracy, accuracy7, accuracy8 = self.naive_bayes_classifier()
        print("#####################################################################")
        print("Total accuracy is {:.2f}%".format(accuracy * 100))
        print("Accuracy for symbol 7 is {:.2f}%".format(accuracy7 * 100))
        print("Accuracy for symbol 8 is {:.2f}%".format(accuracy8 * 100))
        print("#####################################################################")


class LogisticRegression:
    """
    Logistic regression binary classifier
    """

    def __init__(self, train_x, train_y, test_x, test_y):
        """
        :param train_x: train data features
        :param train_y: test data labels
        :param test_x: train data features
        :param text_y: test data labels
        """
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def sigmoid(self, z):
        """
        Calculate sigmoid function of the nparray
        :param z: input array
        :return: sigmoid activation of each element.
        """
        return 1 / (1 + np.exp(-z))

    def train(self, epochs, lr):
        """
        Train the eight as per the training data
        :param epochs: number of iterations to train weights
        :param lr: learning rate to update the weights
        :return: trained weights
        """
        w = np.zeros(self.train_x.shape[1])
        for _ in tqdm(range(epochs)):
            z = np.dot(self.train_x, w)
            predictions = self.sigmoid(z)

            dZ = self.train_y[0] - predictions
            dW = np.dot(self.train_x.T, dZ)
            w += lr * dW

        return w

    def get_prediction_accuracy(self):
        test_y = self.test_y[0]
        count = defaultdict(int)
        predictions = self.get_prediction()
        for idx, prediction in enumerate(predictions):
            if prediction == test_y[idx]:
                count['total'] += 1
                count[prediction] += 1

        print("#####################################################################")
        print("Total accuracy is {:2f}".format(count['total']*100 / len(predictions)))
        print("Accuracy for symbol 7 is {:.2f}%".format(count[0] * 100 / sum(x == 0 for x in test_y)))
        print("Accuracy for symbol 8 is {:.2f}%".format(count[1] * 100 / sum(x == 1 for x in test_y)))
        print("#####################################################################")

    def get_prediction(self):
        trained_weights = self.train(10000, 1e-3)
        predictions = [[0, 1][x > 0.5] for x in self.sigmoid(np.dot(self.test_x, trained_weights))]
        return predictions


if __name__ == "__main__":
    dataset = scipy.io.loadmat('mnist_data.mat')
    train_y = dataset['trY']
    train_x = dataset['trX']
    test_y = dataset['tsY']
    test_x = dataset['tsX']

    naive_bayes_classifier = NaiveBayesClassifier(train_x, train_y, test_x, test_y)
    naive_bayes_classifier.get_naive_bayes_accuracy()

    logistic_classifier = LogisticRegression(train_x, train_y, test_x, test_y)
    logistic_classifier.get_prediction_accuracy()
