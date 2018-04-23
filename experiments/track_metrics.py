from sklearn.neural_network import MLPClassifier
from sacred import Experiment
from ingredients.data import train_data_ingredient, load_data
from ingredients.preproc import preprocess_ingredient, preprocess_data

ex = Experiment('titanic',
                ingredients=[train_data_ingredient, preprocess_ingredient])


@ex.config
def cfg():
    n_epochs = 500


@ex.automain
def run(n_epochs, _run):
    X_train, X_val, Y_train, Y_val = load_data()
    clf = MLPClassifier(warm_start=True, max_iter=1)

    X_train_pp = preprocess_data(X_train)
    X_val_pp = preprocess_data(X_val)

    val_acc_l = []
    for i in range(n_epochs):

        clf.fit(X_train_pp, Y_train)
        train_acc = clf.score(X_train_pp, Y_train)
        val_acc = clf.score(X_val_pp, Y_val)
        val_acc_l.append(val_acc)
        ex.log_scalar("training.accuracy", train_acc, i)
        ex.log_scalar("validation.accuracy", val_acc, i)

    return max(val_acc_l)

