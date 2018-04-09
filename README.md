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

0. Launch local mongo instance: `mongod`
1. Run Experiment (result will be stored in `sacred` database in mongo): `python model_pipeline.py -m sacred`

## See Results

Start local Sacredboard server and connect to local MongoDB instance listening on 27017, database name `sacred`: `sacredboard -m sacred`
