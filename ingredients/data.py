import pandas as pd
from sklearn.model_selection import train_test_split
from sacred import Ingredient


train_data_ingredient = Ingredient('train_dataset')


@train_data_ingredient.config
def cfg():
    filename = 'data/train.csv'
    target = 'Survived'
    split_size = .75


@train_data_ingredient.capture
def load_data(filename, target, split_size):
    data = pd.read_csv(filename)
    features = [i for i in data.columns if i != target]
    return train_test_split(data[features], data[target],
                            train_size=split_size)
