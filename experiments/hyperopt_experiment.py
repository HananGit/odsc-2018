"""
Based on:
1. https://gab41.lab41.org/effectively-running-thousands-of-experiments-hyperopt-with-sacred-dfa53b50f1ec
2. https://github.com/Lab41/pythia/blob/master/experiments/hyperopt_experiments.py
"""

import hyperopt
from hyperopt import fmin, tpe, hp,Trials
from model_accuracy import run as run_titanic, ex as titanic_experiment
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.initialize import Scaffold
import argparse
import numpy as np
import pickle


titanic = 'titantic' #because i misspelled titanic in model_accuracy
def objective(titanic_args):
    """
    :param titantic_args:
    :return:
    """

    # arguments to pass as config_updates dict
    global args
    # result to pass to hyperopt
    global result
    # command-line arguments
    global parse_args


    ex = Experiment('Hyperopt',ingredients=[titanic_experiment])
    ex.observers.append(MongoObserver.create(url=parse_args.mongo_db_address, db_name=parse_args.mongo_db_name))

    args = titanic_args
    ex.main(run_titanic_with_global_args)

    r = ex.run(config_updates=titanic_args)

    #print("Config: ",r.config) #sanity check

    return result


def run_titanic_with_global_args():
    global args
    global result

    try:
        # TODO: will update this later to pass in just the params dealing with a particular model, or the model itself
        # will have to make some changes to experiments/model_accuracy.py, the config and main.
        all_results = run_titanic(args[titanic]['penalty'],args[titanic]['fit_intercept'])

        # For Hyperopt: multiply accuracy * -1 for hyperopt fmin
        result = -all_results

        # For Sacred: Return all_results for sacred
        return all_results
    except:
        # Have sacred log a null result
        return None


def run_titanic_hyperopt():
    # Define the space for titantic search
    # Parameters for each Ingredient
    space = {
        # Set Seed to 0 to make consistent for all HPO runs
        "seed": 0,


        # Data Ingredient: train_dataset
        'train_dataset': {"filename":"data/train.csv",
                            "target": 'Survived',
                            "split_size": .75
                          },
        # Preprocess Ingredient: preprocess
        'preprocess': {
            "features": hp.choice("features",[['Fare','SibSp'],['Fare', 'SibSp', 'Parch']]),
        },

        # Experiment: titanic
        titanic: {
            "fit_intercept":hp.choice('fit_intercept', [True, False]),
            "penalty":hp.choice('penalty', ["l1", "l2"])
        }
    }

    trials = Trials()
    # main hyperopt fmin function
    optimal_run = fmin(objective,
                       space,
                       algo=tpe.suggest,
                       max_evals= parse_args.num_runs,
                       trials=trials)

if __name__ == '__main__':
    """
    Runs a series of test using Hyperopt to determine which parameters are most important
    All tests will be logged using Sacred - if the sacred database is set up correctly, otherwise it will simply run
    To run pass in the number of hyperopt runs, the mongo db address and name

    Generic:
        python experiments/hyperopt_experiments.py NUM_RUNS MONGO_HOST:MONGO_PORT MONGO_SACRED_COLLECTION

    For 5 tests to local mongo instance with collection named sacred:
        python experiments/hyperopt_experiments.py 5 127.0.0.1:27017 sacred

    """


    parser = argparse.ArgumentParser(description="Titanic Hyperopt Tests logging to Sacred")
    parser.add_argument("num_runs", type=int, help="Number of Hyperopt Runs")
    parser.add_argument("mongo_db_address", type=str, help="Address of the Mongo DB")
    parser.add_argument("mongo_db_name", type=str, help="Name of the Mongo DB")

    global parse_args
    parse_args = parser.parse_args()

    if int(parse_args.num_runs)<=0:
        print("Must have more than one run")

    # Monkey patch to avoid having to declare all our variables
    def noop(item):
        pass
    Scaffold._warn_about_suspicious_changes = noop

    trial_results, best = run_titanic_hyperopt()
