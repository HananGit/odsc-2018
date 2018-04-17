import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sacred import Ingredient


data_ingredient = Ingredient('dataset')


@data_ingredient.config
def cfg():
    """Default config"""
    path_train = 'data/train.csv'
    path_val = None
    path_test = 'data/test.csv'
    target = 'Survived'
    split_size = None


@data_ingredient.named_config
def variant_presplit():
    """Predetermined constant split"""
    path_train = 'data/train_presplit.csv'
    path_val = 'data/val_presplit.csv'


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
def load_data(path_train, path_val, path_test, target, split_size=None):
    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)
    feature_cols = [i for i in train_df.columns if i != target]

    ret_d = {}
    if split_size is None or split_size == 1:
        ret_d['train'] = (train_df[feature_cols], train_df[target])
        if path_val:
            val_df = pd.read_csv(path_val)
            ret_d['val'] = (val_df[feature_cols], val_df[target])
    elif 0 < split_size < 1:
        x_train, x_val, y_train, y_val = train_test_split(
            train_df[feature_cols], train_df[target],
            train_size=split_size)
        ret_d['train'] = (x_train, y_train)
        ret_d['val'] = (x_val, y_val)

    else:
        raise ValueError

    ret_d['test'] = (test_df[feature_cols], None)

    return ret_d
