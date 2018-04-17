import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sacred import Ingredient


data_ingredient = Ingredient('dataset')


@data_ingredient.config
def cfg():
    """Default config"""
    path_train = 'data/train.csv'
    path_test = 'data/test.csv'
    target = 'Survived'
    split_size = None


@data_ingredient.named_config
def variant_presplit():
    """Predetermined constant split"""
    path_train = 'data/train_presplit.csv'
    path_test = 'data/val_presplit.csv'


@data_ingredient.named_config
def variant_split():
    """
    Split training set on runtime
    """
    split_size = 0.75


@data_ingredient.named_config
def variant_production():
    """Dummy example of possibly reading from a production
    test file on s3"""
    import datetime
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    path_test = f's3://bucket/date={today}/data.csv'


@data_ingredient.capture
def load_data(path_train, path_test, target, split_size=None):
    train_df = pd.read_csv(path_train)
    feature_cols = [i for i in train_df.columns if i != target]

    ret_d = {}
    if split_size is None or split_size == 1:
        ret_d['train'] = (train_df[feature_cols], train_df[target])
        test_df = pd.read_csv(path_test)
        test_targs = test_df[target] if target in test_df.columns \
            else np.zeros(len(test_df)) * np.nan
        ret_d['test'] = (test_df[feature_cols], test_targs)

    elif 0 < split_size < 1:
        x_train, x_test, y_train, y_test = train_test_split(
            train_df[feature_cols], train_df[target],
            train_size=split_size)
        ret_d['train'] = (x_train, y_train)
        ret_d['test'] = (x_test, y_test)
    else:
        raise ValueError

    return ret_d
