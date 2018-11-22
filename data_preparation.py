import os
import urllib.request
import json
import numpy as np
import pandas as pd

def download_datasets(data_spec):
    if not os.path.exists('data'):
        os.makedirs('data')

    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')

    # download file and convert to npy
    for name, spec in data_spec.items():
        # All of the datasets are text files
        save_file = 'data/raw/{}.txt'.format(name)

        # file exists
        if os.path.isfile(save_file):
            print('Ignore', save_file, ': file exists')
            continue

        url = spec['url']
        try:
            urllib.request.urlretrieve(url, save_file)
        except Exception as e:
            print('Failed to download', save_file)
            raise e
        else:
            print('Succesfully download', save_file)

def convert_to_npz(data_spec):
    print('Start Dataset conversion')
    for name, spec in data_spec.items():
        raw_file = 'data/raw/{}.txt'.format(name)
        save_file = 'data/{}.npz'.format(name)

        # read dataframe
        layout = spec['layout']
        if layout == 'txt':
            df = pd.read_csv(
                    raw_file, 
                    header=None, delim_whitespace=True, engine='python', 
                    **spec['kwargs'])
        elif layout == 'csv':
            df = pd.read_csv(
                    raw_file, 
                    header=None, engine='python', 
                    **spec['kwargs'])
        else:
            raise Exception

        # drop null rows
        df = df[(df.values != spec['null_value']).all(axis=1)]

        # preprocess
        for col, method in spec['preprocess']:
            if method == 'comma_string_to_float':
                df[col] = df[col].replace({',':'.'}, regex=True).astype(float)
            elif method == 'factorize':
                df[col] = pd.factorize(df[col])[0]

        # split target y
        y = df[spec["target"]]

        # drop unused columns
        X = df.drop(columns=[spec["target"]]+spec['drop_cols'])

        # create dummy variables


        # validate
        n, p = X.shape
        K = len(y.unique())
        assert n == spec['n'], ("Dataset size mismatch in {}, "
            "expected n = {}, but got {}".format(name, spec['n'], n))
        assert p == spec['p'], ("Feature dimension mismatch in {}, "
            "expected p = {}, but got {}".format(name, spec['p'], p))
        assert K == spec['K'], ("Number of class labels mismatch in {}, "
            "expected K = {}, but got {}".format(name, spec['K'], K))

        np.savez(save_file, X=X, y=y)
        print("Succesfully saved {}".format(save_file))

def preprocess():
    data_spec = json.load(open('filelist.json'))
    download_datasets(data_spec)
    convert_to_npz(data_spec)

if __name__ == '__main__':
    preprocess()
