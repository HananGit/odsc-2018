from hyperopt import hp

# HPO params for experiments/model_accuracy.py
vanilla_exp_space = {
    # Set Seed to 0 to make consistent for all HPO runs
    "seed": 0,

    # Data Ingredient: train_dataset
    'train_dataset': {
        'filename': 'data/train.csv',
        'target': 'Survived',
        'split_size': .75
    },
    # Preprocess Ingredient: preprocess
    'preprocess': {
        "features": hp.choice("features", [['Fare', 'SibSp'], ['Fare', 'SibSp', 'Parch']]),
    },

    # Experiment: titanic
    'titanic': {
        "fit_intercept": hp.choice('fit_intercept', [True, False]),
        "penalty": hp.choice('penalty', ["l1", "l2"]),
        "C": hp.loguniform('C', .1, 1000)
    }
}

# HPO params for experiments/model_accuracy2.py
stage0_space = {
    # Set Seed to 0 to make consistent for all HPO runs
    "seed": 0,

    # Data Ingredient: train_dataset
    'dataset': {
        'path_train': 'data/train_presplit.csv',
        'path_val': 'data/val_presplit.csv'
    },
    # Preprocess Ingredient: preprocess
    'preprocess': {
        "features": hp.choice("features", [['Fare', 'SibSp'], ['Fare', 'SibSp', 'Parch']]),
    },

    # Experiment: titanic

    "fit_intercept": hp.choice('fit_intercept', [True, False]),
    "penalty": hp.choice('penalty', ["l1", "l2"]),
    "C": hp.loguniform('C', .01, 1000),
    "save_submission": True

}

# HPO params for experiments/model_accuracy3.py
stage0_space_multiple_models = {
    # Set Seed to 0 to make consistent for all HPO runs
    "seed": 0,

    # Data Ingredient: train_dataset
    'dataset': {
        'path_train': 'data/train_presplit.csv',
        'path_val': 'data/val_presplit.csv'
    },
    # Preprocess Ingredient: preprocess
    'preprocess': {
        "features": hp.choice("features", [['Fare', 'SibSp'], ['Fare', 'SibSp', 'Parch']]),
    },

    # Experiment: titanic
    "model_type": hp.choice("model_type", ["lr", "rf"]),

    # LR PARAMS
    "lr_fit_intercept": hp.choice('lr_fit_intercept', [True, False]),
    "lr_penalty": hp.choice('lr_penalty', ["l1", "l2"]),
    "lr_c": hp.loguniform('lr_c', .01, 1000),

    # RF Params
    "rf_n_estimators": hp.choice('rf_n_estimators', [10, 15, 20, 100]),
    "rf_max_depth": hp.choice('rf_max_depth', [None, 5, 10]),
    "rf_min_samples_split": hp.choice('rf_min_samples_split', [2, 5, 10]),

    "save_submission": True

}