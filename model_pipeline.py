from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sacred import Experiment, Ingredient
import pandas as pd
import numpy as np

train_data_ingredient = Ingredient('train_dataset')


@train_data_ingredient.config
def load_data_cfg():
    filename = 'data/train.csv'
    target = 'Survived'
    split_size = .75


@train_data_ingredient.capture
def load_data(filename, target, split_size):
    data = pd.read_csv(filename)
    features = [i for i in data.columns if i != target]
    return train_test_split(data[features], data[target],
                            train_size=split_size)


preprocess_ingredient = Ingredient('preprocess')


@preprocess_ingredient.config
def cfg_preprocess():
    features = ['Fare', 'SibSp', 'Parch']


@preprocess_ingredient.named_config
def variant_preprocess_data():
    features = ['Fare', 'SibSp']


@preprocess_ingredient.capture
def preprocess_data(df, features):
    return df[features]


ex = Experiment('titantic',
                ingredients=[train_data_ingredient, preprocess_ingredient])


@ex.config
def cfg():
    penalty = 'l2'
    fit_intercept = False
    # folds = 10


@ex.automain
def run(penalty, fit_intercept):
    X_train, X_val, Y_train, Y_val = load_data()

    clf_lg = LogisticRegression(penalty=penalty, fit_intercept=fit_intercept)
    clf_lg.fit(preprocess_data(X_train), Y_train)

    return clf_lg.score(preprocess_data(X_val), Y_val)
