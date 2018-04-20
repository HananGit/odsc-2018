from hyperopt import hp

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
        "C": hp.uniform('C', .1, 1000)
    }
}

# Parameters for each Ingredient
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
    "C" : hp.uniform('C',.01,1000),
    "save_submission":True

}

# Parameters for each Ingredient
blender_space = {
    # Set Seed to 0 to make consistent for all HPO runs
    "seed": 0,

    # Data Ingredient: train_dataset
    'dataset': {
        'path_train': None,
        'path_val': None,
        'path_test': None,
        'blended': True
    },
    # Preprocess Ingredient: preprocess
    'preprocess': {
        "features": '*',
    },

    # Experiment: titanic
    "fit_intercept": hp.choice('fit_intercept', [True, False]),
    "penalty": hp.choice('penalty', ["l1", "l2"]),
    "C" : hp.uniform('C',.01,1000)

}
