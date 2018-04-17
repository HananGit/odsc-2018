"""
Creates pre-determined split from training set
Input: data/train.csv
Output: data/train_presplit.csv, data/val_presplit.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    train_df = pd.read_csv('data/train.csv')
    tsplit_df, vsplit_df = train_test_split(
        train_df, test_size=0.25, random_state=0)

    tsplit_df.to_csv('data/train_presplit.csv', index=False, header=True)
    vsplit_df.to_csv('data/val_presplit.csv', index=False, header=True)



