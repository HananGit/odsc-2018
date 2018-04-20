from sacred.observers import MongoObserver
from sacred.initialize import Scaffold
import argparse
from hyperopt import fmin, tpe, hp, Trials


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
        self.base_experiment.observers.append(MongoObserver.create(url=self.mongo_url, db_name=self.mongo_db))

    def objective(self, experiment_args):
        self.experiment_config = experiment_args
        run_obj = self.base_experiment.run(
            config_updates=self.experiment_config)

        return - run_obj.result

    def run_hyperopt(self):
        """
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


def gather_experiments_and_configs(experiment_file_name):
    from experiments.model_accuracy import ex as vanilla
    from experiments.model_accuracy2 import ex as model_params
    from experiments.model_accuracy3 import ex as multiple_models

    from hpo.hyperopt_hpo_configs import vanilla_exp_space, stage0_space, stage0_space_multiple_models

    all_experiments = {
        'model_accuracy': vanilla,
        'model_accuracy2': model_params,
        'model_accuracy3': multiple_models,
    }

    hpo_params = {
        "model_accuracy": vanilla_exp_space,
        "model_accuracy2": stage0_space,
        "model_accuracy3": stage0_space_multiple_models,
    }

    return all_experiments[experiment_file_name], hpo_params[experiment_file_name]


if __name__ == '__main__':
    """
    Runs a series of test using Hyperopt to determine which parameters are most important
    All tests will be logged using Sacred - if the sacred database is set up correctly, otherwise it will simply run
    To run pass in the number of hyperopt runs, the mongo db address and name

    Generic:
        python hpo/hyperopt_hpo.py NUM_RUNS MONGO_HOST:MONGO_PORT MONGO_SACRED_COLLECTION EXPERIMENT_FILE_NAME

    For 5 tests to local mongo instance with collection named sacred:
        python hpo/hyperopt_hpo.py 5 127.0.0.1:27017 sacred model_accuracy

    """

    parser = argparse.ArgumentParser(description="Titanic Hyperopt Tests logging to Sacred")
    parser.add_argument("num_runs", type=int, help="Number of Hyperopt Runs")
    parser.add_argument("mongo_db_address", type=str, help="Address of the Mongo DB")
    parser.add_argument("mongo_db_name", type=str, help="Name of the Mongo DB")
    parser.add_argument("experiment_file_name", type=str,
                        help="Which hpo params to used. Add new in hpo/hyperopt_hpo_configs.py")

    global parse_args
    parse_args = parser.parse_args()

    if int(parse_args.num_runs) <= 0:
        print("Must have more than one run")

    # Monkey patch to avoid having to declare all our variables
    def noop(item):
        pass
    Scaffold._warn_about_suspicious_changes = noop

    base_experiment, space = gather_experiments_and_configs(parse_args.experiment_file_name)

    hyperopt_exps = HyperoptHPO(base_experiment=base_experiment,
                                command_line_args=parse_args,
                                param_space=space)

    trial_results, best = hyperopt_exps.run_hyperopt()
