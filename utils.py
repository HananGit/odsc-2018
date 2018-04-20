import pandas as pd
from pymongo import MongoClient
from ingredients.data_smoothie import artifact_by_name
import gridfs

def gather_predictions(run_id):
    """Get predictions from mongo given a data.blend run_id"""

    client = MongoClient('localhost', 27017)
    db = client['sacred']
    fs = gridfs.GridFS(db)
    collection = db.runs

    query = collection.find({"_id":run_id})
    for doc in query:
        preds_obj_id = artifact_by_name(doc, "test_submission")
        df = pd.read_csv(fs.get(preds_obj_id)) \
            .set_index('PassengerId') \
            .rename({'pred_proba': f"pred_proba-{doc['_id']}"}, axis=1)
        df.to_csv(f"data/gen_sub_{run_id}.csv", header=True, index=True)
        break

# gather_predictions(489)