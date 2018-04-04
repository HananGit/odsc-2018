from numpy.random import permutation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sacred import Experiment
import pandas as pd

ex = Experiment('titantic')

@ex.config
def cfg():
  penalty = 'l2'
  fit_intercept = False
  features = ['Fare','SibSp','Parch']
  folds = 10

@ex.automain
def run(penalty, fit_intercept,features,folds):

	train = pd.read_csv('data/train.csv')

	X_train = train[features]
	Y_train = train[['Survived']]

	clf_lg = LogisticRegression(penalty=penalty,fit_intercept=fit_intercept)
	clf_lg.fit(X_train,Y_train)

	return cross_val_score(clf_lg,X_train,Y_train,cv=folds).mean()