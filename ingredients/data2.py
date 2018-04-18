import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sacred import Ingredient
from ingredients.data_smoothie import gather_stage0_features


data_ingredient = Ingredient('dataset')


@data_ingredient.config
def cfg():
    """Default config"""
    path_train = 'data/train.csv'
    path_val = None
    path_test = 'data/test.csv'
    index_col = 'PassengerId'
    target_col = 'Survived'
    split_size = None

    blended = False


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


@data_ingredient.named_config
def blend():
    """Blend of predictions from our top 3 models
    Example:
        ```
        python experiments/model_accuracy2.py -m sacred with \
        variant_rand_params dataset.blend preprocess.variant_all \
        save_submission=True
        ```
    """
    path_train = None
    path_val = None
    path_test = None
    blended = True


@data_ingredient.capture
def load_data(path_train, path_val, path_test,
              index_col, target_col,
              split_size=None,
              blended=False, ):

    if blended:
        return gather_stage0_features(target_col)

    train_df = pd.read_csv(path_train, index_col=index_col)
    test_df = pd.read_csv(path_test, index_col=index_col)
    feature_cols = [i for i in train_df.columns if i != target_col]

    ret_d = {}
    if split_size is None or split_size == 1:
        ret_d['train'] = (train_df[feature_cols], train_df[target_col])
        if path_val:
            val_df = pd.read_csv(path_val, index_col=index_col)
            ret_d['val'] = (val_df[feature_cols], val_df[target_col])
    elif 0 < split_size < 1:
        x_train, x_val, y_train, y_val = train_test_split(
            train_df[feature_cols], train_df[target_col],
            train_size=split_size)
        ret_d['train'] = (x_train, y_train)
        ret_d['val'] = (x_val, y_val)

    else:
        raise ValueError

    ret_d['test'] = (test_df[feature_cols], None)

    return ret_d
