import pytest
import numpy as np
import pandas as pd
from src.utils.split_train_test import stratified_split, split_train_test
from skmultilearn.model_selection import iterative_train_test_split

@pytest.mark.skip(reason="Speedup tests by ignoring")
def test_stratified_split():
    np.random.seed(1)

    y = np.array(
        [['sad', '29f'], ['sad', '29f'], ['sad', '28m'], ['ang', '28m'], ['ang', '01m'], ['unt', '01m'], ['fea', '01m'],
         ['fea', '28m'], ['fea', '23m']])
    X = np.random.randint(0, 10, (len(y), 5))
    emotions = {'ang': 0, 'fea': 1, 'hap': 2, 'neu': 3, 'sad': 4, 'unt': 5}
    y = np.array([np.array([emotions[labels[0]], int(labels[1][:-1])]) for labels in y])
    folds, classes = stratified_split(X, y, 2)
    print('Folds are:')
    print(folds)
    print('Classes are:')
    print(classes)

    ''' TODO: go through each class, check that difference between number of rows per emotion between folds is <= 1, same for subject '''

@pytest.mark.skip(reason="Speedup tests by ignoring")
def test_stratified_split():
    y = np.array([[0, 1], [0, 3], [1, 3], [4, 5], [4, 3], [4, 4], [4, 4]])
    X = np.array([[i, i + 1] for i in range(len(y))])
    assert len(X) == len(y)
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = classes.shape[0]

    class_counts = np.bincount(y_indices)
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.5)
    print('X_train')
    print(X_train)
    print('y_train')
    print(y_train)
    print('X_test')
    print(X_test)
    print('y_test')
    print(y_test)
    # check no. counts of [4,4] isn't 2 in test

def test_strat_train_test_split():
    df = pd.DataFrame(data={'emotion':['ang', 'fea', 'ang', 'hap', 'sad', 'sad'],
                            'subject':['1m', '1m', '2f', '2f', '2f', '2f'],
                            'x': [0,1,2,3,4,5]
                            })
    # df = pd.DataFrame(data={'emotion':['ang', 'fea', 'unt', 'hap', 'fea', 'sad'],
    #                         'subject':['1m', '1m', '2f', '2f', '2f', '2f'],
    #                         'x': [0,1,2,3,4,5]
    #                         })
    train, test = split_train_test(df, 0.5, 0.5)
    print('train')
    print(train)
    print('test')
    print(test)
