import csv
import json
import operator
import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn import svm
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from matplotlib import pyplot as plt

from sklearn.datasets import load_breast_cancer


import numpy as np

titles = []

def normalizing(data_set=[]):

    scalar = StandardScaler()
    scalar.fit(data_set)

    standardized_data = scalar.transform(data_set)

    minmax_data = minmax_scale(data_set)

    plt.subplot(2, 2, 1)
    plt.title('Raw Data')
    plt.boxplot(data_set)

    # plt.figure()
    plt.subplot(2, 2, 2)
    plt.title('MinMax Data')
    plt.boxplot(minmax_data)

    plt.subplot(2, 2, 3)
    plt.title('Standardized Data')
    plt.boxplot(standardized_data)

    plt.show()

    return standardized_data


def csv_reader():
    global titles
    input_data = pd.read_csv('final_data_in.csv', index_col=0)  # import data
    values = input_data.values

    titles = input_data.axes[1].values[1:]
    col_values = values[:, 2:]  # select all columns from column 2 onwards

    scaled_data = normalizing(col_values)

    return scaled_data


def evaluate_pca_number(scaled_data=[]):
    """Evaluate pca for different number of Components
    Return: Number of components (Integer)"""

    pca = PCA()
    pca.fit(scaled_data)
    cum_sum = np.cumsum(pca.explained_variance_ratio_)
    components = len(cum_sum)

    plt.plot(cum_sum)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    for i in range(len(cum_sum)):
        if cum_sum[i] >= 0.9:
            components = i
            break

    return components


def principle_component_analysis(p_components=0, scaled_data=[]):
    pca = PCA(n_components=p_components)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)

    pp = pca.components_
    print(pp)

    return x_pca


def dimension_reduction_pca(scaled_data=[]):
    # cancer = load_breast_cancer()
    # cc = cancer['feature_names']

    output = scaled_data[:, 0:1]
    inputs = scaled_data[:, 1:]

    p_components = evaluate_pca_number(scaled_data=inputs)  # Draw graph by evaluating pca

    x_pca = principle_component_analysis(p_components, inputs)

    # plt.figure(figsize=(8, 6))
    # based = input_data[['out_mig_index']]
    # based = based.values[:, 0]  # Select 1st column of the ndarray
    # based = np.array(based)
    # plt.scatter(x_pca[:, 0], x_pca[:, 1], c=based, cmap='plasma')
    # plt.show()

    ls3 = [list(l1) + list(l2) for l1, l2 in zip(output, x_pca)]

    return ls3


def k_fold_validation(clf=None, X=[], y=[]):
    k_fold = KFold(n_splits=10)

    t0 = time.time()
    for train_index, test_index in k_fold.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        print("Score: ", score)
    print("Time Taken: ", time.time() - t0)


def support_vector(X=[], y=[]):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for kernel in kernels:
        print("Train using Support Vector Machine kernel: {0}".format(kernel))
        clf = svm.SVR(kernel)
        k_fold_validation(clf, X, y)


def kernel_ridge(X=[], y=[]):
    print("Train Using Kernel ridge")
    clf = KernelRidge()
    k_fold_validation(clf, X, y)


def grid_search(X=[], y=[], regression='svr'):
    print("Running GridSearch {0}".format(regression))
    clf = None
    parameters = None
    if regression == 'svr':
        clf = svm.SVR(kernel='rbf')
        parameters = {"C": [1e0, 1e1], "gamma": np.logspace(-2, 2, 5)}
    elif regression == 'rf':
        clf = RandomForestRegressor(n_jobs=4)
        parameters = {'n_estimators': [5, 20], 'max_features': ['auto', 'sqrt', 'log2']}
    elif regression == 'kr':
        clf = KernelRidge(kernel='rbf', gamma=0.1)
        parameters = {"alpha": [1e0, 0.1], "gamma": np.logspace(-2, 2, 5)}
    clf = GridSearchCV(clf, parameters, n_jobs=4)
    clf.fit(X, y)
    # results = clf.cv_results_.keys()
    print(clf.best_params_)
    k_fold_validation(clf, X, y)


def random_forest(X=[], y=[]):
    print("Train using RandomForest")
    regr = RandomForestRegressor()
    k_fold_validation(regr, X, y)


def feature_importance(X=[], y=[], labels=[]):
    print('Running the feature importance')
    # Split the data into 40% test and 60% training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    # Create a random forest classifier

    clf = RandomForestRegressor(n_estimators=10000, random_state=0, n_jobs=-1)

    # Train the classifier
    clf.fit(X_train, y_train)

    importance = {}
    # Print the name and gini importance of each feature
    for label, feature in zip(labels, clf.feature_importances_):
        importance[label] = feature

    sorted_x = sorted(importance.items(), key=operator.itemgetter(1))
    print(sorted_x)

if __name__ == '__main__':

    # Read the csv and standardise it and return
    scaled_data1 = csv_reader()
    inputs = scaled_data1[:, 1:]
    outputs = scaled_data1[:, 0]

    # Feature Selection
    # feature_importance(inputs, outputs, titles[1:])

    # Support vector machine
    # With pca
    x = dimension_reduction_pca(scaled_data1)
    x = np.array(x)
    support_vector(x[:, 1:], x[:, 0])

    print("Regressions without pca data")
    # without pca
    support_vector(inputs, outputs)

    # Random Forest Regression

    # inputs = x[:, 1:]
    # outputs = x[:, 0]
    random_forest(inputs, outputs)

    # Grid Search SVR
    grid_search(inputs, outputs, regression='svr')

    # Grid Search RandomForest
    grid_search(inputs, outputs, regression='rf')

    # Grid Search KernelRidge
    grid_search(inputs, outputs, regression='kr')

