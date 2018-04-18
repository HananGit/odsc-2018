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
    C = 1.0
    save_probs = True
    save_submission = False


@ex.named_config
def variant_rand_params():
    penalty = np.random.choice(['l1', 'l2'])
    fit_intercept = np.random.randint(2, dtype=bool)
    C = np.exp(np.random.randn() * 5)


def df_artifact(ex, df, name=None):
    """Writes a DataFrame as an artifact (csv format)"""
    f_tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    df.to_csv(f_tmp, header=True, index=True)
    f_tmp.close()
    ex.add_artifact(f_tmp.name, name=name)
    os.remove(f_tmp.name)
    return f_tmp.name


@ex.automain
def run(penalty, fit_intercept, C,
        save_probs, save_submission):
    data_d = load_data()
    x_train, y_train = data_d['train']
    x_val, y_val = data_d.get('val', (None, None))
    x_test, y_test = data_d.get('test', (None, None))

    clf_lg = LogisticRegression(
        penalty=penalty, fit_intercept=fit_intercept, C=C,
    )

    clf_lg.fit(preprocess_data(x_train), y_train)

    # Test Predictions
    if x_test is not None:
        pred_prob_test = clf_lg.predict_proba(preprocess_data(x_test))
        pred_lbl_test = pred_prob_test.argmax(axis=1)

        # Export prob predictions
        if save_probs:
            # Val
            prob_df = pd.DataFrame(
                pred_prob_test[:, 1],
                index=pd.Index(x_test.index, name='PassengerId'),
                columns=['pred_proba'])
            prob_df['Survived'] = y_test
            df_artifact(ex, prob_df, 'test_predictions')

        # Export submission
        if save_submission:
            sub_df = pd.DataFrame(
                pred_lbl_test,
                index=pd.Index(x_test.index, name='PassengerId'),
                columns=['Survived'])
            df_artifact(ex, sub_df, 'test_submission')

    # Validation Predictions
    if not (x_val is None or y_val is None):
        pred_prob_val = clf_lg.predict_proba(preprocess_data(x_val))
        pred_lbl_val = pred_prob_val.argmax(axis=1)
        score_val = accuracy_score(y_val, pred_lbl_val)

        # Export prob predictions
        if save_probs:
            # Val
            prob_df = pd.DataFrame(
                pred_prob_val[:, 1],
                index=pd.Index(x_val.index, name='PassengerId'),
                columns=['pred_proba'])
            prob_df['Survived'] = y_val
            df_artifact(ex, prob_df, 'holdout_predictions')

        return score_val

    else:

        return None

