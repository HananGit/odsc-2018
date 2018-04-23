from sklearn.linear_model import SGDClassifier
from sacred import Experiment
from ingredients.data import train_data_ingredient, load_data
from ingredients.preproc import preprocess_ingredient, preprocess_data
import numpy as np

ex = Experiment('titanic',
                ingredients=[train_data_ingredient, preprocess_ingredient])


@ex.config
def cfg():
    penalty = 'l1'
    learning_rate = 'optimal'
    n_epochs = 10


@ex.automain
def run(penalty, learning_rate, n_epochs, _run):
    X_train, X_val, Y_train, Y_val = load_data()

    clf_lg = SGDClassifier(penalty=penalty, learning_rate=learning_rate, max_iter=1, shuffle=True)

    for i in range(n_epochs):

        clf_lg.partial_fit(preprocess_data(X_train), Y_train, classes=np.unique(Y_train))
        _run.log_scalar("training.accuracy", clf_lg.score(preprocess_data(X_train), Y_train), i)
        _run.log_scalar("validation.accuracy", clf_lg.score(preprocess_data(X_val), Y_val), i)
