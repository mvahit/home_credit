# LightGBM GBDT with KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
import gc
import os
import pickle
import pandas as pd

from lightgbm import LGBMClassifier
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from scripts.helper_functions import display_importances


def kfold_lightgbm(df, debug=False):
    # Divide in training/validation and test data

    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()

    folds = KFold(n_splits=10, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])  # predicted valid_y
    sub_preds = np.zeros(test_df.shape[0])  # submission preds
    feature_importance_df = pd.DataFrame()  # feature importance

    fold_auc_best_df = pd.DataFrame(columns=["FOLD", "AUC", "BEST_ITER"])  # holding best iter to save model
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index',
                                                      "APP_index", "BURO_index", "PREV_index", "INSTAL_index",
                                                      "CC_index", "POS_index"]]

    # folds split'e X,Y birlikte gösterildi. Bu veriyi bol dendi. 10 tane train-validasyon index cifti turetildi.
    # enumerate turetilen index çiftlerini çift olarak yakalama imkanı sagladi.

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            n_jobs=-1,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x,
                train_y,
                eval_set=[(train_x, train_y),
                          (valid_x, valid_y)],
                eval_metric='auc',
                verbose=200,
                early_stopping_rounds=200)

        # predicted valid_y
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        # submission preds. her kat icin test setini tahmin edip tum katların ortalamasini alıyor.
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        # fold, auc and best iteration
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        # best auc & iteration
        fold_auc_best_df = fold_auc_best_df.append({'FOLD': int(n_fold + 1),
                                                    'AUC': roc_auc_score(valid_y, oof_preds[valid_idx]),
                                                    "BEST_ITER": clf.best_iteration_}, ignore_index=True)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    # OUTPUTS
    print(fold_auc_best_df)
    print(feature_importance_df)

    # feature importance'ları df olarak kaydet
    feature_importance_df.to_pickle("outputs/features/feature_importance_df.pkl")
    fold_auc_best_df.to_pickle("outputs/features/fold_auc_best_df.pkl")

    # Final Model
    best_iter_1 = int(fold_auc_best_df.sort_values(by="AUC", ascending=False)[:1]["BEST_ITER"].values)

    # AUC'ye gore sırala, ilk 3 fold'un best iter sayılarının ortalamasını al, virgulden sonra sayı olmasın.
    # best_iter_3 = round(fold_auc_best_df.sort_values(by="AUC", ascending=False)[:3]["BEST_ITER"].mean(), 0)

    y_train = train_df["TARGET"]
    x_train = train_df[feats]

    final_model = LGBMClassifier(
            n_jobs=-1,
            n_estimators=best_iter_1,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1).fit(x_train, y_train)

    cur_dir = os.getcwd()
    os.chdir('models/reference/')
    pickle.dump(final_model, open("lightgbm_final_model.pkl", 'wb'))  # model
    os.chdir(cur_dir)

    # her bir fold icin tahmin edilen valid_y'ler aslında train setinin y'lerinin farklı parcalarda yer alan tahminleri.
    print('Full Train(Validasyon) AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv("predictions/reference_submission.csv", index=False)

    display_importances(feature_importance_df)
    del x_train, y_train

    return feature_importance_df

