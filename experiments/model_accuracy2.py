import uuid
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sacred import Experiment
from ingredients.data2 import data_ingredient, load_data
from ingredients.preproc import preprocess_ingredient, preprocess_data


ex = Experiment('titantic',
                ingredients=[data_ingredient, preprocess_ingredient])


@ex.config
def cfg():
    """
    See the following doc for possible string values of `metric`
    http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    Make sure to turn on `predict_proba` if the metric requires probabilities
    """
    penalty = 'l2'
    fit_intercept = False
    save_probs = True
    save_submission = False


@ex.automain
def run(penalty, fit_intercept, save_probs, save_submission):
    data_d = load_data()
    x_train, y_train = data_d['train']
    x_test, y_test = data_d['test']

    clf_lg = LogisticRegression(penalty=penalty, fit_intercept=fit_intercept)

    clf_lg.fit(preprocess_data(x_train), y_train)

    pred_prob_test = clf_lg.predict_proba(preprocess_data(x_test))
    pred_lbl_test = pred_prob_test.argmax(axis=1)

    score = accuracy_score(y_test, pred_lbl_test)
    # score = roc_auc_score(y_test, pred_prob_test[:, 1])

    # Export predictions
    if save_probs:
        path_save_probs = f'./saved/{str(uuid.uuid4())}.npy'
        np.save(path_save_probs, pred_prob_test[:, 1])
        ex.add_artifact(path_save_probs, name='predictions')

    if save_submission:
        sub_df = pd.DataFrame(
            pred_lbl_test,
            index=pd.Index(x_test.index, name='PassengerId'),
            columns=['Survived'])
        path_save_sub = f'./saved/{str(uuid.uuid4())}.csv'
        sub_df.to_csv(path_save_sub, header=True, index=True)
        ex.add_artifact(path_save_sub, name='submission')

    return score

