import os
import numpy as np
import pandas as pd
import tempfile
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
    model_type = "lr"  # LogisticRegression or SVM

    # LR Params
    lr_penalty = 'l2'
    lr_fit_intercept = False
    lr_c = 1.0

    # SVM Params
    svm_c = 1.0
    svm_kernel = 'rbf'  # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    svm_degree = 3
    svm_gamma = 'rbf'  # ‘rbf’, ‘poly’, ‘sigmoid’, 'auto'
    svm_probability = True

    save_probs = True
    save_submission = False


@ex.named_config
def variant_rand_params():

    model_type = np.random.choice(['lr','svm'])

    lr_penalty = np.random.choice(['l1', 'l2'])
    lr_fit_intercept = np.random.randint(2, dtype=bool)
    lr_c = np.exp(np.random.randn() * 5)

    svm_kernel = 'linear'  # np.random.choice(['linear', 'poly', 'sigmoid', 'rbf'])
    svm_gamma = 'auto'  # np.random.choice(['auto', 'poly', 'sigmoid', 'rbf'])
    svm_degree = np.random.randint(3) + 1
    svm_c = np.exp(np.random.randn() * 5)


def df_artifact(ex, df, name=None):
    """Writes a DataFrame as an artifact (csv format)"""
    f_tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    df.to_csv(f_tmp, header=True, index=True)
    f_tmp.close()
    ex.add_artifact(f_tmp.name, name=name)
    os.remove(f_tmp.name)
    return f_tmp.name


@ex.automain
def run(lr_penalty,
        lr_fit_intercept,
        lr_c,
        svm_kernel,
        svm_gamma,
        svm_degree,
        svm_c,
        svm_probability,
        model_type,
        save_probs,
        save_submission):

    data_d = load_data()
    x_train, y_train = data_d['train']
    x_val, y_val = data_d.get('val', (None, None))
    x_test, y_test = data_d.get('test', (None, None))

    clf = None
    if model_type == 'lr':
        clf = LogisticRegression(
            penalty=lr_penalty, fit_intercept=lr_fit_intercept, C=lr_c
        )
    elif model_type == 'svm':
        clf = SVC(
            C=svm_c, kernel=svm_kernel, gamma=svm_gamma,
            degree=svm_degree, probability=svm_probability
        )
    else:
        print("model type " + model_type + " not defined")

    try:
        clf.fit(preprocess_data(x_train), y_train)
    except TypeError:
        print("kernel: " + str(svm_kernel))
        print("degree: " + str(svm_degree))
        print("gamma: " + str(svm_gamma))

    # Test Predictions
    if x_test is not None:
        pred_prob_test = clf.predict_proba(preprocess_data(x_test))
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
        pred_prob_val = clf.predict_proba(preprocess_data(x_val))
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

