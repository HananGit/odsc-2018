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
We defined a `variant_simple` in our 
[preprocessing ingredient](ingredients/preproc.py). To run this variant:
```python model_pipeline.py with preprocess.variant_simple seed=0```

Similarly, we have a config option called `save_submission` which is `False`
by default. We can turn it on from the CLI, which causes a submission file
to be generated and tracked as an artifact.
```python model_pipeline.py with seed=0 save_submission=True```

We can even use `print_config` to show a dry run of config and what's changed
from the default
```python experiments/model_accuracy2.py print_config with seed=0 dataset.variant_split save_submission=True```

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