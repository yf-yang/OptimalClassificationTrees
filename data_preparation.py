import os
import urllib.request
import json
import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool

def download_file(name, spec):
    url = spec['url']

    # All of the datasets are ascii files, so save with extension .txt
    save_file = 'data/raw/{}.txt'.format(name)

    # file exists
    if os.path.isfile(save_file):
        print('Ignore', save_file, ': file exists')

    try:
        urllib.request.urlretrieve(url, save_file)
    except Exception as e:
        print('Got', type(e).__name__, '-', e,
            '\nIgnore', save_file, ': fail to download')
    else:
        print('Succesfully download', save_file)

def download_datasets(data_spec):
    if not os.path.exists('data'):
        os.makedirs('data')

    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')

    # download files
    with Pool(16) as pool:
        pool.starmap(download_file, data_spec.items())
        
def convert_to_npz(data_spec):
    print()
    print('-----------------------------------')
    print('Dataset Conversion')
    for name, spec in data_spec.items():
        raw_file = 'data/raw/{}.txt'.format(name)
        save_file = 'data/{}.npz'.format(name)

        # read dataframe
        layout = spec['layout']
        try:
            if layout == 'csv':
                df = pd.read_csv(
                        raw_file, 
                        header=None, engine='python', 
                        **spec['kwargs'])
            elif layout == 'txt':
                df = pd.read_csv(
                        raw_file, 
                        header=None, delim_whitespace=True, engine='python', 
                        **spec['kwargs'])
            elif layout == "special":
                df = pd.read_csv(raw_file, engine='python', **spec['kwargs'])
        except FileNotFoundError:
            print('Ignore', raw_file, ': file not found on disk')
            continue

        # drop null rows
        df = df[(df.values != spec['null_value']).all(axis=1)]

        # preprocess
        for col, method in spec['preprocess']:
            if method == 'comma_string_to_float':
                df[col] = df[col].replace({',':'.'}, regex=True).astype(float)

        # split target y
        y = pd.factorize(df[spec["target"]])[0]

        # drop unused columns
        X = df.drop(columns=[spec["target"]]+spec['drop_cols'])

        # create dummy variables
        if spec['dummy_cols']:
            assert spec["target"] not in spec['dummy_cols']
            dummies = pd.get_dummies(X[spec['dummy_cols']].astype(str), drop_first=True)
            non_dummies = X.drop(columns=spec['dummy_cols']) 
            X = pd.concat([non_dummies, dummies], axis=1)

        # validate dataset settings
        n, p = X.shape
        K = len(np.unique(y))
        assert n == spec['n'], ("Dataset size mismatch in {}, "
            "expect n = {}, but got {}".format(name, spec['n'], n))
        assert p == spec['p'], ("Feature dimension mismatch in {}, "
            "expect p = {}, but got {}".format(name, spec['p'], p))
        assert K == spec['K'], ("Number of class labels mismatch in {}, "
            "expect K = {}, but got {}".format(name, spec['K'], K))

        # Convert to float
        X = X.values.astype(np.float32)
        y = y.astype(np.int)

        np.savez(save_file, X=X, y=y)
        print("Succesfully save {}".format(save_file))

def preprocess():
    data_spec = json.load(open('filelist.json'))
    download_datasets(data_spec)
    convert_to_npz(data_spec)

if __name__ == '__main__':
    preprocess()
