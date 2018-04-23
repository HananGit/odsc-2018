from sklearn.linear_model import LogisticRegression
from sacred import Experiment
from ingredients.data import train_data_ingredient, load_data
from ingredients.preproc import preprocess_ingredient, preprocess_data


ex = Experiment('titanic',
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
