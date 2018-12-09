import numpy as np
from sklearn.model_selection import train_test_split

def run(file, ratio, model, num_round = 1, seed=None, settings = {}):
    train_ratio, val_ratio, test_ratio = map(int, ratio.split())
    train_perc = train_ratio / (train_ratio+val_ratio+test_ratio)
    test_perc = test_ratio / (val_ratio+test_ratio)

    # load dataset
    data = np.load(file)
    X = data['X']
    y = data['y']

    # split dataset 
    X_train, X_test_val, y_train, y_test_val = train_test_split(
        X, y, train_size=train_perc, random_state=seed)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test_val, y_test_val, train_size=test_perc, random_state=seed)

    model.fit(X_train, y_train, **settings)
    