import os
import numpy as np
import pandas as pd
import tempfile
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
        f_tmp = tempfile.NamedTemporaryFile(delete=False)
        np.save(f_tmp, pred_prob_test[:, 1])
        f_tmp.close()
        ex.add_artifact(f_tmp.name, name='predictions')
        os.remove(f_tmp.name)

    if save_submission:
        f_tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        sub_df = pd.DataFrame(
            pred_lbl_test,
            index=pd.Index(x_test.index, name='PassengerId'),
            columns=['Survived'])
        sub_df.to_csv(f_tmp, header=True, index=True)
        f_tmp.close()
        ex.add_artifact(f_tmp.name, name='submission')
        os.remove(f_tmp.name)

    return score

