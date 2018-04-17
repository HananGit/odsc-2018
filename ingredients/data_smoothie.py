import pandas as pd
import pymongo
from pymongo import MongoClient
import gridfs


def artifact_by_name(doc, name):
    """Convenience fn to grab ObjectId by artifact name"""
    obj_id = [d for d in doc['artifacts']
              if d['name'] == name][0]['file_id']
    return obj_id


def gather_stage0_features(target):
    """Blend of predictions from our top 3 models"""

    client = MongoClient('localhost', 27017)
    db = client.sacred
    fs = gridfs.GridFS(db)
    collection = db.runs

    # Note: normally, we would normally enforce some sort of diversity between
    # top models
    #     ex) they must be different types of classifiers, or something

    query = collection.find() \
        .sort('result', pymongo.DESCENDING) \
        .limit(3)

    train_df_l = []
    test_df_l = []
    for doc in query:
        for df_l, name in [(train_df_l, 'holdout_predictions'),
                           (test_df_l, 'test_predictions')]:

            preds_obj_id = artifact_by_name(doc, name)
            df = pd.read_csv(fs.get(preds_obj_id)) \
                .set_index('PassengerId') \
                .rename({'pred_proba': f"pred_proba-{doc['_id']}"}, axis=1)
            df_l.append(df)

    train_df = pd.concat(train_df_l, axis=1)
    # De-dupe redundant target columns in training
    train_df = train_df.loc[:, ~train_df.columns.duplicated()]
    test_df = pd.concat(test_df_l, axis=1)

    ret_d = {'train': (train_df.drop(target, axis=1), train_df[target]),
             'test': (test_df.drop(target, axis=1), None)}

    return ret_d
