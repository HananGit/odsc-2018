# Experimental Reproducibility in Data Science with Sacred

Demonstrate how to use Sacred to track machine learning experiments on popular kaggle titanic competition

## Getting Started

- Kaggle Titanic Competition
- Sacred

### Prerequisites

- python3 - https://www.python.org/downloads/
- mongo - https://docs.mongodb.com/manual/installation/

### Setup

0. Clone repo: `git clone https://github.com/gilt/odsc-2018`
1. Setup Virtual Environment: `python3 -m venv /PATH/TO/odsc-2018/sacred-demo`
2. Install python packages: `pip install -r requirements.txt`

## Run Experiment

### Running an experiment with all defaults
`python model_pipeline.py`
Notice that different runs yield different results since we have not  
controlled the random seed.

But if we fix the seed by running:
`python model_pipeline.py seed=0`
we should end up with the same results on every run.

### Running a variant

#### Variants in experiment
To run a different variant of our experiment:
```python experiments/model_accuracy2.py print_config with variant_rand_params```

Similarly, we have a config option called `save_submission` which is `False`
by default. We can turn it on from the CLI, which causes a submission file
to be generated and tracked as an artifact.
```python model_pipeline.py with seed=0 save_submission=True```

#### Variants in ingredients
We also defined a `variant_simple` in our
[preprocessing ingredient](ingredients/preproc.py). To run this variant:
```python model_pipeline.py with preprocess.variant_simple seed=0```

We can even use `print_config` to show a dry run of config and what's changed
from the default
```python experiments/model_accuracy2.py print_config with seed=0 dataset.variant_split save_submission=True```

#### Vary a bunch of stuff
```python experiments/model_accuracy2.py with variant_rand_params save_submission=True dataset.variant_presplit```


### Running with a mongo observer
0. Launch local mongo instance: `mongod`
1. Run Experiment (result will be stored in `sacred` database in mongo): 
`python model_pipeline.py -m sacred`

## See Results

### Mongo
To look at all our runs on mongo:
```
mongo
use sacred
db.runs.find().pretty()
```

### Sacredboard
Start local Sacredboard server and connect to local MongoDB instance listening on 27017, database name `sacred`: `sacredboard -m sacred`


# References
- [sacred](https://github.com/IDSIA/sacred) [(pub)](http://ml.informatik.uni-freiburg.de/papers/17-SciPy-Sacred.pdf)
- [sacredboard](https://github.com/chovanecm/sacredboard)
- [Kaggle Titanic](https://www.kaggle.com/c/titanic)


## Model Blending Workflow

Run the following to simulate various experiments with random parameter search:
```bash
python experiments/model_accuracy2.py -m sacredblender \
    with variant_rand_params dataset.variant_presplit save_submission=True
```
Note: we switched to a new database `sacredblender` in case there's any
garbage in the `sacred` database. This is required because we've hardcoded
the lookup to `sacredblender` database

Now run an experiment that blends the top 3 runs based on holdout performance:
```bash
python experiments/model_accuracy2.py -m sacred \
    with dataset.blend preprocess.variant_all save_submission=True
```
Note: we switched back to `sacred` database here. We also went with the
default parameters for the meta blender model