"""modele train ya test bağımsız değişken değerlerini sor"""

import os
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='prediction_type', action='store_true')
parser.add_argument('--test', dest='prediction_type', action='store_false')
parser.set_defaults(prediction_type=True)
args = parser.parse_args()

final_train = pd.read_pickle("data/final_train_df.pkl")
final_test = pd.read_pickle("data/final_test_df.pkl")

feats = [f for f in final_test.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index',
                                                    "APP_index", "BURO_index", "PREV_index", "INSTAL_index",
                                                    "CC_index", "POS_index"]]

if args.prediction_type:
    y_train = final_train["TARGET"]
    x_train = final_train[feats]

    cur_dir = os.getcwd()
    os.chdir('models/reference/')
    model = pickle.load(open('lightgbm_final_model.pkl', 'rb'))
    os.chdir(cur_dir)

    y_pred = model.predict_proba(x_train)[:, 1]
    print("TRAIN AUC SCORE:", roc_auc_score(y_train, y_pred))
else:
    x_test = final_test[feats]
    cur_dir = os.getcwd()
    os.chdir('models/reference/')
    model = pickle.load(open('lightgbm_final_model.pkl', 'rb'))
    os.chdir(cur_dir)
    y_pred = model.predict_proba(x_test)[:, 1]
    ids = final_test['SK_ID_CURR']
    submission = pd.DataFrame({'SK_ID_CURR': ids, 'TARGET': y_pred})
    os.chdir('outputs/predictions/')
    submission.to_csv("sub_from_prediction_py.csv", index=False)
    print("Submission file has been created in:", "/Users/mvahit/Documents/GitHub/home_credit/predictions/")

# calistirmak icin
# python scripts/predict.py --train
