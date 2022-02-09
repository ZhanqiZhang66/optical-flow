#!/usr/bin/python3
# Homework 4 Code
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt

#%%
def bagged_tree(X_train, y_train, X_test, y_test, num_bags):
    # The `bagged_tree` function learns an ensemble of numBags decision trees using a random subset of the features
    # at each split on the input dataset and also plots the  out-of-bag error as a function of the number of bags
    #
    # % Inputs:
    # % * `X_train` is the training data
    # % * `y_train` are the training labels
    # % * `X_test` is the testing data
    # % * `y_test` are the testing labels
    # % * `num_bags` is the number of trees to learn in the ensemble
    #
    # % Outputs:
    # % * `out_of_bag_error` is the out-of-bag classification error of the final learned ensemble
    # % * `test_error` is the classification error of the final learned ensemble on test data
    #
    # % Note: You may use sklearns 'DecisonTreeClassifier'
    # but **not** 'RandomForestClassifier' or any other bagging function

    out_of_bag_predictions = np.empty([X_train.shape[0], num_bags])
    out_of_bag_predictions[:] = np.nan
    test_predictions = np.empty([X_test.shape[0], num_bags])
    test_predictions[:] = np.nan

    out_of_bag_errors = []

    for i in range(num_bags):
        in_bag_indicies = np.random.choice(X_train.shape[0], X_train.shape[0])
        out_of_bag_indicies = np.setdiff1d(np.arange(0, X_train.shape[0]), in_bag_indicies)

        X_in_bag = X_train[in_bag_indicies, :]
        y_in_bag = y_train[in_bag_indicies]

        X_out_bag = X_train[out_of_bag_indicies, :]
        y_out_bag = y_train[out_of_bag_indicies]

        decision_tree = DecisionTreeClassifier(criterion='entropy')
        # print(X_in_bag, y_in_bag)
        decision_tree.fit(X_in_bag, y_in_bag)

        out_of_bag_predictions[out_of_bag_indicies, i] = decision_tree.predict(X_out_bag)
        test_predictions[:, i] = decision_tree.predict(X_test)

        aggregate_out_of_bag_predictions = mode(out_of_bag_predictions[:, 0:i + 1], axis=1)[0]
        valid_predictions = ~np.isnan(aggregate_out_of_bag_predictions).reshape(-1)

        out_of_bag_errors.append(np.sum(
            aggregate_out_of_bag_predictions[valid_predictions].reshape(-1) != np.array(
                y_train[valid_predictions])) / len(
            y_train[valid_predictions]))

    plt.plot(np.arange(1, num_bags+1), out_of_bag_errors)
    plt.xlabel('Number of bags')
    plt.ylabel('Out-of-bag error')
    plt.title('bagged tree')
    plt.show()

    out_of_bag_error = out_of_bag_errors[-1]
    aggregate_test_predictions = mode(test_predictions, axis=1)[0]
    test_error = np.sum(aggregate_test_predictions.reshape(-1) != y_test) / X_test.shape[0]

    return out_of_bag_error, test_error


def single_decision_tree(X_train, y_train, X_test, y_test):
    # The `single_decision_tree` function is the basic decision tree model
    #
    # % Inputs:
    # % * `X_train` is the training data
    # % * `y_train` are the training labels
    # % * `X_test` is the testing data
    # % * `y_test` are the testing labels
    #
    # % Outputs:
    # % * `train_error` is the classification error of decision tree on test data
    # % * `test_error` is the classification error of decision tree on test data

    tree_model = DecisionTreeClassifier(criterion='entropy')
    # best_tree = DecisionTreeClassifier(max_features=d_max)
    tree_model.fit(X_train, y_train)
    train_prediction = tree_model.predict(X_train)
    train_error = np.sum(train_prediction != y_train) / X_test.shape[0]
    test_prediction = tree_model.predict(X_test)
    test_error = np.sum(test_prediction != y_test) / X_test.shape[0]

    return  train_error, test_error


def main_hw4(num_bags=200):
    print("number of bags is ", num_bags)

    # Load data
    og_train_data = np.genfromtxt('zip.train.txt')
    og_test_data = np.genfromtxt('zip.test.txt')

    # Find ones and threes
    print(" one_vs_three ")
    one_vs_three = np.where(og_train_data[:, 0] == 1)[0].tolist() + np.where(og_train_data[:, 0] == 3)[0].tolist()
    train_data = og_train_data[one_vs_three, :]
    one_vs_three = np.where(og_test_data[:, 0] == 1)[0].tolist() + np.where(og_test_data[:, 0] == 3)[0].tolist()
    test_data = og_test_data[one_vs_three, :]

    # Split data
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]


    # Run random forest
    out_of_bag_errors = []
    test_errors = []
    for i in range(0, 1):
        out_of_bag_error, test_error = bagged_tree(X_train, y_train, X_test, y_test, num_bags)
        out_of_bag_errors.append(out_of_bag_error)
        test_errors.append(test_error)

    print('The range of out-of-bag error for the bagged tree is between %.4f and %.4f' % (
        np.min(out_of_bag_errors), np.max(out_of_bag_errors)))
    print('The range of test error for the bagged tree is between %.4f and %.4f' % (
        np.min(test_errors), np.max(test_errors)))

    train_error, test_error = single_decision_tree(X_train, y_train, X_test, y_test)
    print("train error of decision tree is ", train_error)
    print("test error of decision tree is ", test_error)

    # Find threes and fives
    print(" three_vs_five ")
    five_vs_three = np.where(og_train_data[:, 0] == 5)[0].tolist() + np.where(og_train_data[:, 0] == 3)[0].tolist()
    train_data = og_train_data[five_vs_three, :]
    five_vs_three = np.where(og_test_data[:, 0] == 5)[0].tolist() + np.where(og_test_data[:, 0] == 3)[0].tolist()
    test_data = og_test_data[five_vs_three, :]

    # Split data
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]


    # Run random forest
    out_of_bag_errors = []
    test_errors = []
    for i in range(0, 1):
        out_of_bag_error, test_error = bagged_tree(X_train, y_train, X_test, y_test, num_bags)
        out_of_bag_errors.append(out_of_bag_error)
        test_errors.append(test_error)

    print('The range of out-of-bag error for the bagged tree is between %.4f and %.4f' % (
        np.min(out_of_bag_errors), np.max(out_of_bag_errors)))
    print('The range of test error for the bagged tree is between %.4f and %.4f' % (
        np.min(test_errors), np.max(test_errors)))

    train_error, test_error = single_decision_tree(X_train, y_train, X_test, y_test)
    print("train error of decision tree is ", train_error)
    print("test error of decision tree is ", test_error)


def main_cv():
    # Load data
    og_train_data = np.genfromtxt('zip.train.txt')
    og_test_data = np.genfromtxt('zip.test.txt')

    # Find ones and threes
    print(" one_vs_three ")
    one_vs_three = np.where(og_train_data[:, 0] == 1)[0].tolist() + np.where(og_train_data[:, 0] == 3)[0].tolist()
    train_data = og_train_data[one_vs_three, :]
    one_vs_three = np.where(og_test_data[:, 0] == 1)[0].tolist() + np.where(og_test_data[:, 0] == 3)[0].tolist()
    test_data = og_test_data[one_vs_three, :]

    # Split data
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]

    # Set parameters
    m = int(np.floor(X_train.shape[1] / 3))
    cv = 10
    cross_val_score, test_error = single_decision_tree(X_train, y_train, X_test, y_test, m, cv)
    print("cross val error of decision tree is ", 1 - cross_val_score)
    print("test error of decision tree is ", test_error)


if __name__ == "__main__":
    np.random.seed(153)
    num_bags = 200
    main_hw4(num_bags)