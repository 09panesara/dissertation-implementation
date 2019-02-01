import pytest
import numpy as np
from src.utils.split_train_test import stratified_split

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
