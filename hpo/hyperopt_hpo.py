from experiments.model_accuracy import ex as titanic_experiment
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.initialize import Scaffold
import argparse
from hyperopt import fmin, tpe, hp, Trials


titanic = 'titantic'  # Misspelled titanic in model_accuracy


class HyperoptHPO(object):
    """
    Runs a hyperopt experiment over a Sacred Experiment
    """
    # replaces the "args" global
    experiment_config = None
    # replaces the "result" global
    result = None

    def __init__(self, base_experiment, command_line_args, param_space):
        """Create a new HyperoptHPO object to run HPO over a Sacred experiment

        :param base_experiment: Experiment, required
        :param command_line_args: args, required
        :param param_space: dict, required
        """
        self.base_experiment = base_experiment
        # replace the parse_args global
        self.mongo_url = command_line_args.mongo_db_address
        self.mongo_db = command_line_args.mongo_db_name
        self.num_runs = command_line_args.num_runs
        self.param_space = param_space

        # initialize Experiment
        self.hyperopt_exp = Experiment('Hyperopt',
                                       ingredients=[self.base_experiment])
        self.hyperopt_exp.observers.append(
            MongoObserver.create(url=self.mongo_url, db_name=self.mongo_db))

    def objective(self, experiment_args):
        self.experiment_config = experiment_args
        run_obj = self.base_experiment.run(
            config_updates=self.experiment_config)

        # print("Config: ",r.config) #sanity check

        return - run_obj.result

    def run_hyperopt(self):
        """
        Replaces run_titanic_hyperopt
        :return: trials: performance on each trial run, and optimal_run: the best run
        """

        trials = Trials()
        # main hyperopt fmin function
        optimal_run = fmin(
            self.objective,
            self.param_space,
            algo=tpe.suggest,
            max_evals=self.num_runs,
            trials=trials)

        return trials, optimal_run


if __name__ == '__main__':
    """
    Runs a series of test using Hyperopt to determine which parameters are most important
    All tests will be logged using Sacred - if the sacred database is set up correctly, otherwise it will simply run
    To run pass in the number of hyperopt runs, the mongo db address and name

    Generic:
        python hpo/hyperopt_hpo.py NUM_RUNS MONGO_HOST:MONGO_PORT MONGO_SACRED_COLLECTION

    For 5 tests to local mongo instance with collection named sacred:
        python experiments/hyperopt_experiment.py 5 127.0.0.1:27017 sacred

    """

    parser = argparse.ArgumentParser(description="Titanic Hyperopt Tests logging to Sacred")
    parser.add_argument("num_runs", type=int, help="Number of Hyperopt Runs")
    parser.add_argument("mongo_db_address", type=str, help="Address of the Mongo DB")
    parser.add_argument("mongo_db_name", type=str, help="Name of the Mongo DB")

    global parse_args
    parse_args = parser.parse_args()

    if int(parse_args.num_runs) <= 0:
        print("Must have more than one run")

    # Monkey patch to avoid having to declare all our variables
    def noop(item):
        pass
    Scaffold._warn_about_suspicious_changes = noop

    # Parameters for each Ingredient
    space = {
        # Set Seed to 0 to make consistent for all HPO runs
        "seed": 0,

        # Data Ingredient: train_dataset
        'train_dataset': {
            "filename": "data/train.csv",
            "target": 'Survived',
            "split_size": .75
        },
        # Preprocess Ingredient: preprocess
        'preprocess': {
            "features": hp.choice("features", [['Fare', 'SibSp'], ['Fare', 'SibSp', 'Parch']]),
        },

        # Experiment: titanic
        titanic: {
            "fit_intercept": hp.choice('fit_intercept', [True, False]),
            "penalty": hp.choice('penalty', ["l1", "l2"])
        }
    }

    hyperopt_exps = HyperoptHPO(base_experiment=titanic_experiment,
                                command_line_args=parse_args,
                                param_space=space)

    trial_results, best = hyperopt_exps.run_hyperopt()
