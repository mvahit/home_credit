import gc
import re
import time
import warnings
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
import seaborn as sns
from feature_engine import categorical_encoders as ce
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore')


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# Display plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :100].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    print(best_features)
    plt.figure(figsize=(15, 20))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


def load_dataset(file_path, index=0):
    df = pd.read_csv(file_path, index_col=index)
    return df


def get_categoric_columns(df):
    cols = df.select_dtypes(include=['object', 'category']).columns
    return cols


def apply_label_encoding(l_df, columns):
    lbe = LabelEncoder()
    for col in columns:
        l_df[col] = lbe.fit_transform(l_df[col])
    return l_df


def apply_one_hot_encoding(l_df):
    original_columns = list(l_df)  # col names as string in a list
    categorical_columns = get_categoric_columns(l_df)  # categorical col names
    l_df = pd.get_dummies(l_df, columns=categorical_columns, drop_first=True)  # creating dummies
    new_columns = [c for c in l_df.columns if c not in original_columns]  # new col names
    return l_df, new_columns


def rare_encoding(data, variables, rare_threshold=0.05, n_rare_categories=4):
    encoder = ce.RareLabelCategoricalEncoder(tol=rare_threshold, n_categories=n_rare_categories, variables=variables,
                                             replace_with='Rare')
    # fit the encoder
    encoder.fit(data)
    # transform the data
    data = encoder.transform(data)
    return data


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns





def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
            to reduce memory usage.
        """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    # code takenn from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    return df


def feature_early_shutdown(row):
    early_shutdown = 0
    if row.CREDIT_ACTIVE == "Active" and row.DAYS_CREDIT_ENDDATE < 0:
        early_shutdown = 1
    return early_shutdown


def buro_add_feature(df_breau):
    df_bureau_new = pd.DataFrame()
    # kredi başvuru sayısı
    df_bureau_new["BURO_CREDIT_APPLICATION_COUNT"] = df_breau.groupby("SK_ID_CURR").count()["SK_ID_BUREAU"]

    # aktif kredi sayısı
    df_bureau_new["BURO_ACTIVE_CREDIT_APPLICATION_COUNT"] = \
        df_breau[df_breau["CREDIT_ACTIVE"] == "Active"].groupby("SK_ID_CURR").count()["CREDIT_ACTIVE"]
    df_bureau_new["BURO_ACTIVE_CREDIT_APPLICATION_COUNT"].fillna(0, inplace=True)

    # pasif kredi sayısı
    df_bureau_new["BURO_CLOSED_CREDIT_APPLICATION_COUNT"] = \
        df_breau[df_breau["CREDIT_ACTIVE"] == "Closed"].groupby("SK_ID_CURR").count()["CREDIT_ACTIVE"]
    df_bureau_new["BURO_CLOSED_CREDIT_APPLICATION_COUNT"].fillna(0, inplace=True)

    # erken kredi kapama
    df_bureau_new["BURO_EARLY_SHUTDOWN_NEW"] = df_breau.apply(lambda x: feature_early_shutdown(x), axis=1)

    # geciktirilmiş ödeme sayısı
    df_bureau_new["BURO_NUMBER_OF_DELAYED_PAYMENTS"] = \
        df_breau[df_breau["AMT_CREDIT_MAX_OVERDUE"] != 0].groupby("SK_ID_CURR")["AMT_CREDIT_MAX_OVERDUE"].count()
    df_bureau_new["BURO_NUMBER_OF_DELAYED_PAYMENTS"].fillna(0, inplace=True)

    # son kapanmış başvurusu üzerinden geçen max süre
    df_bureau_new["BURO_MAX_TIME_PASSED_CREDIT_APPLICATION"] = \
        df_breau[df_breau["CREDIT_ACTIVE"] == "Closed"].groupby("SK_ID_CURR")["DAYS_ENDDATE_FACT"].max()
    df_bureau_new["BURO_MAX_TIME_PASSED_CREDIT_APPLICATION"].fillna(0, inplace=True)

    # geciktirilmiş max ödeme tutari
    df_bureau_new["BURO_MAX_DELAYED_PAYMENTS"] = df_breau.groupby("SK_ID_CURR")["AMT_CREDIT_MAX_OVERDUE"].max()
    df_bureau_new["BURO_MAX_DELAYED_PAYMENTS"].fillna(0, inplace=True)

    # geciktirilmiş ödeyenlerden oluşan top liste - en yüksek 100
    # gecikme olan (80302, 12)
    df_bureau_new["BURO_DELAYED_PAYMENTS_TOP_100_NEW"] = \
        df_bureau_new.sort_values("BURO_MAX_DELAYED_PAYMENTS", ascending=False)["BURO_MAX_DELAYED_PAYMENTS"].rank()
    df_bureau_new["BURO_DELAYED_PAYMENTS_TOP_100_NEW"].fillna(0, inplace=True)

    # kredi uzatma yapilmis mi
    df_bureau_new["BURO_IS_CREDIT_EXTENSION_NEW"] = df_breau.groupby("SK_ID_CURR")["CNT_CREDIT_PROLONG"].count().apply(
        lambda x: 1 if x > 0 else 0)

    # max yapilan kredi uzatmasi
    df_bureau_new["BURO_CREDIT_EXTENSION_MAX"] = df_breau.groupby("SK_ID_CURR")["CNT_CREDIT_PROLONG"].max()
    df_bureau_new["BURO_CREDIT_EXTENSION_MAX"].fillna(0, inplace=True)

    # unsuccessful credit payment - borç takarak kapanmış kredi ödemeleri tespit et
    df_bureau_new["BURO_IS_UNSUCCESSFUL_CREDIT_PAYMENT_NEW"] = \
        df_breau[(df_breau["CREDIT_ACTIVE"] == "Closed") & (df_breau["AMT_CREDIT_SUM_DEBT"] > 0)].groupby(
            "SK_ID_CURR").all()["AMT_CREDIT_SUM_DEBT"].apply(lambda x: 1 if x == True else 0)
    df_bureau_new["BURO_IS_UNSUCCESSFUL_CREDIT_PAYMENT_NEW"].fillna(0, inplace=True)

    return df_bureau_new


def load_data_with_application_train(num_rows=None):
    df_app_train = application_train()
    print("application_train df shape:", df_app_train.shape)
    bureau, bureau_add_features = bureau_and_balance()
    print("Bureau df shape:", bureau.shape)
    bureau = bureau.fillna(0)
    return df_app_train, bureau, bureau_add_features


def load_data_only_bureau_and_bureau_balance(num_rows=None):
    bureau, bureau_add_features = bureau_and_balance()
    print("Bureau df shape:", bureau.shape)
    bureau = bureau.fillna(0)
    return bureau, bureau_add_features


def app_train_bureau_merge(num_rows=None):
    df_app_train, bureau, bureau_add_features = load_data_with_application_train(num_rows)
    # df_merge = pd.merge(df_app_train, bureau, on=['SK_ID_CURR'],how='inner')
    df_merge = bureau
    # print("app_train, bureau merge shape:", df_merge.shape)
    print("bureau merge shape:", df_merge.shape)
    df_final = pd.merge(df_merge, bureau_add_features, on=['SK_ID_CURR'], how='inner')
    print("Bureau add features df shape:", bureau_add_features.shape)
    del df_app_train, bureau, bureau_add_features, df_merge
    gc.collect()
    return df_final


def bureau_and_bureau_balance_features(num_rows=None):
    bureau, bureau_add_features = load_data_only_bureau_and_bureau_balance(num_rows)
    df_final = pd.merge(bureau, bureau_add_features, on=['SK_ID_CURR'], how='inner')
    print("Bureau add features df shape:", bureau_add_features.shape)
    del bureau, bureau_add_features
    gc.collect()
    return df_final


def application_train():
    conn = pymysql.connect(host='35.228.28.142', port=int(63306), user='group2', passwd='123654', db='home_credit')
    df_app_train = pd.read_sql_query("SELECT * FROM application_train", conn)
    df_app_train = df_app_train[["TARGET", "SK_ID_CURR"]]
    # df_app_train = df_app_train.dropna()
    df_app_train.reset_index(drop=True, inplace=True)
    gc.collect()
    return df_app_train


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(nan_as_category=True):
    conn = pymysql.connect(host='35.228.28.142', port=int(63306), user='group2', passwd='123654', db='home_credit')
    bureau = pd.read_sql_query("SELECT * FROM bureau", conn)
    bb = pd.read_sql_query("SELECT * FROM bureau_balance", conn)
    bureau["AMT_CREDIT_SUM_DEBT"] = bureau["AMT_CREDIT_SUM_DEBT"].fillna(0)
    bureau.fillna(0, inplace=True)
    bb.fillna(0, inplace=True)
    # bureau = bureau.dropna()
    bureau.reset_index(drop=True, inplace=True)
    # bb = bb.dropna()
    bb.reset_index(drop=True, inplace=True)

    # add_features
    bureau_add_features = buro_add_feature(df_breau=bureau)

    # sum agg b_balance
    # Status_sum ile ilgili yeni bir degisken olusturma
    bb_dummy = pd.get_dummies(bb, dummy_na=True)
    agg_list = {"MONTHS_BALANCE": "count",
                "STATUS_0": ["sum"],
                "STATUS_1": ["sum"],
                "STATUS_2": ["sum"],
                "STATUS_3": ["sum"],
                "STATUS_4": ["sum"],
                "STATUS_5": ["sum"],
                "STATUS_C": ["sum"],
                "STATUS_X": ["sum"]}
    bb_sum_agg = bb_dummy.groupby("SK_ID_BUREAU").agg(agg_list)
    # Degisken isimlerinin yeniden adlandirilmasi
    bb_sum_agg.columns = pd.Index(["BURO_" + col[0] + "_" + col[1].upper() for col in bb_sum_agg.columns.tolist()])
    # Status_sum ile ilgili yeni bir degisken olusturma
    bb_sum_agg['BURO_NEW_STATUS_SCORE'] = bb_sum_agg['BURO_STATUS_1_SUM'] + bb_sum_agg['BURO_STATUS_2_SUM'] ^ 2 + \
                                          bb_sum_agg['BURO_STATUS_3_SUM'] ^ 3 + bb_sum_agg['BURO_STATUS_4_SUM'] ^ 4 + \
                                          bb_sum_agg['BURO_STATUS_5_SUM'] ^ 5
    bb_sum_agg.drop(
        ['BURO_STATUS_1_SUM', 'BURO_STATUS_2_SUM', 'BURO_STATUS_3_SUM', 'BURO_STATUS_4_SUM', 'BURO_STATUS_5_SUM'],
        axis=1, inplace=True)

    # CREDIT_TYPE degiskeninin sinif sayisini 3'e düsürmek
    bureau['CREDIT_TYPE'] = bureau['CREDIT_TYPE'].replace(['Car loan',
                                                           'Mortgage',
                                                           'Microloan',
                                                           'Loan for business development',
                                                           'Another type of loan',
                                                           'Unknown type of loan',
                                                           'Loan for working capital replenishment',
                                                           "Loan for purchase of shares (margin lending)",
                                                           'Cash loan (non-earmarked)',
                                                           'Real estate loan',
                                                           "Loan for the purchase of equipment",
                                                           "Interbank credit",
                                                           "Mobile operator loan"], 'Rare')

    # CREDIT_ACTIVE degiskeninin sinif sayisini 2'ye düsürmek (Sold' u Closed a dahil etmek daha mi uygun olur ???)
    bureau['CREDIT_ACTIVE'] = bureau['CREDIT_ACTIVE'].replace(['Bad debt', 'Sold'], 'Active')

    # one hot encoding start
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    # one hot encoding end

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size'],
                       "STATUS_0": ["mean"],
                       "STATUS_C": ["mean"],
                       "STATUS_X": ["mean"]}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']

    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])

    # b_balance sum değşkenlerinin eklenmesi
    bb_agg["BURO_MONTHS_BALANCE_COUNT"] = bb_sum_agg["BURO_MONTHS_BALANCE_COUNT"]
    bb_agg["BURO_STATUS_0_SUM"] = bb_sum_agg["BURO_STATUS_0_SUM"]
    bb_agg["BURO_STATUS_C_SUM"] = bb_sum_agg["BURO_STATUS_C_SUM"]
    bb_agg["BURO_STATUS_X_SUM"] = bb_sum_agg["BURO_STATUS_X_SUM"]
    bb_agg["BURO_NEW_STATUS_SCORE"] = bb_sum_agg["BURO_NEW_STATUS_SCORE"]

    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')

    bureau["BURO_MONTHS_BALANCE_COUNT"].fillna(0, inplace=True)
    bureau["BURO_STATUS_0_SUM"].fillna(0, inplace=True)
    bureau["BURO_STATUS_C_SUM"].fillna(0, inplace=True)
    bureau["BURO_STATUS_X_SUM"].fillna(0, inplace=True)
    bureau["BURO_NEW_STATUS_SCORE"].fillna(0, inplace=True)

    ##ek son değişkenler
    # ortalama kac aylık kredi aldıgını gösteren yeni degisken
    bureau["BURO_NEW_MONTHS_CREDIT"] = round((bureau.DAYS_CREDIT_ENDDATE - bureau.DAYS_CREDIT) / 30)

    bureau.drop(columns='SK_ID_BUREAU', inplace=True)

    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum', 'std'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum', 'std', 'median'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
        'AMT_ANNUITY': ['max', 'mean'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['BURO_ACT_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['BURO_CLS_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg, bureau_add_features


def application_train_g():
    conn = pymysql.connect(host='35.228.28.142', port=int(63306), user='group2', passwd='123654', db='home_credit')
    df = pd.read_sql_query("SELECT * FROM application_train", conn)
    test_df = pd.read_sql_query("SELECT * FROM application_test", conn)

    df = reduce_mem_usage(df)
    test_df = reduce_mem_usage(test_df)

    df = df.append(test_df).reset_index()

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)

    le = LabelEncoder()

    df["NAME_EDUCATION_TYPE"] = le.fit_transform(df["NAME_EDUCATION_TYPE"])
    df.loc[(df["NAME_EDUCATION_TYPE"] == 1), "NAME_EDUCATION_TYPE"] = 0

    df.loc[(df["CNT_FAM_MEMBERS"] > 3), "CNT_FAM_MEMBERS"] = 4

    df = df[df['CODE_GENDER'] != 'XNA']

    lbe = LabelEncoder()

    for col in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[col] = lbe.fit_transform(df[col])

    # df = pd.get_dummies(df, dummy_na = True)

    nom_list = [
        'EMERGENCYSTATE_MODE',
        'FONDKAPREMONT_MODE',
        'HOUSETYPE_MODE',
        'NAME_CONTRACT_TYPE',
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE',
        'NAME_INCOME_TYPE',
        'NAME_TYPE_SUITE',
        'OCCUPATION_TYPE',
        'ORGANIZATION_TYPE',
        'WALLSMATERIAL_MODE',
        'WEEKDAY_APPR_PROCESS_START']

    df = rare_encoding(df, nom_list)
    df = pd.get_dummies(df, columns=nom_list, drop_first=True)

    # new_features
    # 1
    df["APP_NEW_GOODSPRICE/CREDIT"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    # 2
    df["APP_NEW_ANNUITY/CREDIT"] = (df["AMT_ANNUITY"] / df["AMT_CREDIT"])
    # 3
    df["APP_NEW_INCOME/ANNUITY"] = df["AMT_INCOME_TOTAL"] / df["AMT_ANNUITY"]
    # 4
    df["APP_NEW_DAYS_LAST_PHONE_CHANGE"] = df["DAYS_LAST_PHONE_CHANGE"]
    df.loc[(df["APP_NEW_DAYS_LAST_PHONE_CHANGE"] == 0), "APP_NEW_DAYS_LAST_PHONE_CHANGE"] = 1
    df.loc[(df["APP_NEW_DAYS_LAST_PHONE_CHANGE"] != 0), "APP_NEW_DAYS_LAST_PHONE_CHANGE"] = 0
    # 5
    df["DAYS_BIRTH"] = df["DAYS_BIRTH"] / 365
    df["APP_NEW_DAYS_BIRTH"] = df["DAYS_BIRTH"]
    df.loc[(df["APP_NEW_DAYS_BIRTH"] <= 41), "APP_NEW_DAYS_BIRTH"] = 1
    df.loc[(df["APP_NEW_DAYS_BIRTH"] > 41), "APP_NEW_DAYS_BIRTH"] = 0
    # 6
    df["APP_NEW_CREDIT/INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    # 7
    df["APP_NEW_WORK/NOTWORK"] = df["DAYS_EMPLOYED"]
    df.loc[(df["APP_NEW_WORK/NOTWORK"] == 0), "APP_NEW_WORK/NOTWORK"] = 0  # ÇALIŞMAYANLAR
    df.loc[(df["APP_NEW_WORK/NOTWORK"] != 0), "APP_NEW_WORK/NOTWORK"] = 1  # ÇALIŞANLAR
    # 8
    df["APP_NEW_INCOME/CREDIT"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    # 9
    # En yakın zaman (soruşturma olmayan 0, saat+gün+ hafta+ay için 1, ay+yıl için 2)
    df["APP_NEW_REQ"] = df["AMT_REQ_CREDIT_BUREAU_WEEK"]
    # yakın ve orta zamanda soruşturma
    df.loc[(df["AMT_REQ_CREDIT_BUREAU_HOUR"] > 0), "APP_NEW_REQ"] = 1
    df.loc[(df["AMT_REQ_CREDIT_BUREAU_DAY"] > 0), "APP_NEW_REQ"] = 1

    df.loc[(df["AMT_REQ_CREDIT_BUREAU_HOUR"] == 0) & (df["AMT_REQ_CREDIT_BUREAU_DAY"] == 0) & (
            df["AMT_REQ_CREDIT_BUREAU_WEEK"] > 0), "APP_NEW_REQ"] = 1
    df.loc[(df["AMT_REQ_CREDIT_BUREAU_HOUR"] == 0) & (df["AMT_REQ_CREDIT_BUREAU_DAY"] == 0) & (
            df["AMT_REQ_CREDIT_BUREAU_MON"] > 0), "APP_NEW_REQ"] = 1
    # uzak zaman soruşturma
    df.loc[(df["AMT_REQ_CREDIT_BUREAU_HOUR"] == 0) & (df["AMT_REQ_CREDIT_BUREAU_DAY"] == 0) &
           (df["AMT_REQ_CREDIT_BUREAU_WEEK"] == 0) & (df["AMT_REQ_CREDIT_BUREAU_MON"] == 0) &
           (df["AMT_REQ_CREDIT_BUREAU_QRT"] > 0), "APP_NEW_REQ"] = 2

    df.loc[(df["AMT_REQ_CREDIT_BUREAU_HOUR"] == 0) & (df["AMT_REQ_CREDIT_BUREAU_DAY"] == 0) &
           (df["AMT_REQ_CREDIT_BUREAU_WEEK"] == 0) & (df["AMT_REQ_CREDIT_BUREAU_MON"] == 0) &
           (df["AMT_REQ_CREDIT_BUREAU_YEAR"] > 0), "APP_NEW_REQ"] = 2
    # soruşturma olmayanlar
    df.loc[(pd.isna(df["APP_NEW_REQ"])), "APP_NEW_REQ"] = 0

    # eski grup yeni feature ları
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['NEW_DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    df['NEW_EXT_RESOURCE_3_CREDIT_TO_GOODS_RATIO'] = df['EXT_SOURCE_3'] / (df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'])
    df['NEW_EXT_RESOURCE_2_CREDIT_TO_GOODS_RATIO'] = df['EXT_SOURCE_2'] / (df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'])
    df['NEW_EXT_RESOURCE_1_CREDIT_TO_GOODS_RATIO'] = df['EXT_SOURCE_1'] / (df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'])

    df.drop("index", axis=1, inplace=True)

    df.columns = pd.Index(["APP_" + col for col in df.columns.tolist()])

    df.rename(columns={"APP_SK_ID_CURR": "SK_ID_CURR"}, inplace=True)

    df.rename(columns={"APP_TARGET": "TARGET"}, inplace=True)

    return df


def previous_application():
    conn = pymysql.connect(host='35.228.28.142', port=int(63306), user='group2', passwd='123654', db='home_credit')
    df_prev = pd.read_sql_query("SELECT * FROM previous_application", conn)
    df_prev = reduce_mem_usage(df_prev)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    df_prev = df_prev.sample(1000)

    # Features that has outliers
    feat_outlier = ["AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE",
                    "SELLERPLACE_AREA"]

    # Replacing the outliers of the features with their own upper values
    for var in feat_outlier:
        Q1 = df_prev[var].quantile(0.01)
        Q3 = df_prev[var].quantile(0.99)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df_prev[var][(df_prev[var] > upper)] = upper

    # 365243 value will be replaced by NaN in the following features
    feature_replace = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE',
                       'DAYS_TERMINATION']

    for var in feature_replace:
        df_prev[var].replace(365243, np.nan, inplace=True)

    # One hot encoding
    categorical_columns = [col for col in df_prev.columns if df_prev[col].dtype == 'object']
    df_prev = pd.get_dummies(df_prev, columns=categorical_columns, dummy_na=True)

    # Creating new features

    df_prev['APP_CREDIT_PERC'] = df_prev['AMT_APPLICATION'] / df_prev['AMT_CREDIT']
    df_prev['NEW_CREDIT_TO_ANNUITY_RATIO'] = df_prev['AMT_CREDIT'] / df_prev['AMT_ANNUITY']
    df_prev['NEW_DOWN_PAYMENT_TO_CREDIT'] = df_prev['AMT_DOWN_PAYMENT'] / df_prev['AMT_CREDIT']
    df_prev['NEW_TOTAL_PAYMENT'] = df_prev['AMT_ANNUITY'] * df_prev['CNT_PAYMENT']
    df_prev['NEW_TOTAL_PAYMENT_TO_AMT_CREDIT'] = df_prev['NEW_TOTAL_PAYMENT'] / df_prev['AMT_CREDIT']
    # Innterest ratio previous application (simplified)

    df_prev['SIMPLE_INTERESTS'] = (df_prev['NEW_TOTAL_PAYMENT'] / df_prev['AMT_CREDIT'] - 1) / df_prev['CNT_PAYMENT']

    # Previous applications numeric features
    num_aggregations = {}
    num_cols = df_prev.select_dtypes(exclude=['object'])
    num_cols.drop(['SK_ID_PREV', 'SK_ID_CURR'], axis=1, inplace=True)

    for num in num_cols:
        num_aggregations[num] = ['min', 'max', 'mean', 'var', 'sum']

        # Previous applications categoric features
    cat_aggregations = {}
    for i in df_prev.columns:
        if df_prev[i].dtypes == "O":
            cat_aggregations[i] = ['mean']

    prev_agg = df_prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Dropping features with small variance
    features_with_small_variance = prev_agg.columns[(prev_agg.std(axis=0) < .1).values]
    prev_agg.drop(features_with_small_variance, axis=1, inplace=True)
    prev_agg.reset_index(inplace=True)

    return prev_agg


def credit_card_balance():
    conn = pymysql.connect(host='35.228.28.142', port=int(63306), user='group2', passwd='123654', db='home_credit')
    ccb = pd.read_sql_query("SELECT * FROM credit_card_balance", conn)

    ccb = reduce_mem_usage(ccb)
    ccb = ccb.sample(1000)

    ccb = ccb.groupby('SK_ID_CURR').agg(['mean'])
    e = 0
    ccb.columns = pd.Index(
        ['CC_' + ccb.columns[e][0] + "_" + ccb.columns[e][1].upper() for e in range(ccb.columns.size)])

    # new feature1: calculating the rate of balance(loan) to the credit card limit
    ccb["CC_NEW_LOAN_TO_CREDIT_LIMIT_RATE"] = (ccb["CC_AMT_BALANCE_MEAN"] + 1) / (
            ccb["CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN"] + 1)

    # new feature2: at what rate the customer paid the loan:CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN /
    # CC_AMT_TOTAL_RECEIVABLE_MEAN: CC_PAID_AMOUNT_RATE
    ccb["CC_NEW_PAID_AMOUNT_RATE"] = (ccb["CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN"] + 1) / (
            ccb["CC_AMT_TOTAL_RECEIVABLE_MEAN"] + 1) * 100

    # new feature3: how much money the customer withdrew in avg from ATM per drawing:AMOUNT PER ATM DRAWING
    ccb["CC_NEW_AMT_PER_ATM_DRAWING_MEAN"] = (ccb["CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"] + 1) / (
            ccb["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] + 1)

    # new feature4: how much money the customer withdrew from POS in avg per drawing:AMOUNT PER POS DRAWING
    ccb["CC_NEW_AMT_PER_POS_DRAWING_MEAN"] = (ccb["CC_AMT_DRAWINGS_POS_CURRENT_MEAN"] + 1) / (
            ccb["CC_CNT_DRAWINGS_POS_CURRENT_MEAN"] + 1)

    ccb = pd.concat([ccb.loc[:, "CC_NEW_LOAN_TO_CREDIT_LIMIT_RATE"],
                     ccb.loc[:, "CC_NEW_PAID_AMOUNT_RATE"],
                     ccb.loc[:, "CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN"],
                     ccb.loc[:, "CC_AMT_PAYMENT_CURRENT_MEAN"],
                     ccb.loc[:, "CC_MONTHS_BALANCE_MEAN"],
                     ccb.loc[:, "CC_CNT_INSTALMENT_MATURE_CUM_MEAN"],
                     ccb.loc[:, "CC_AMT_INST_MIN_REGULARITY_MEAN"],
                     ccb.loc[:, "CC_AMT_DRAWINGS_ATM_CURRENT_MEAN"],
                     ccb.loc[:, "CC_AMT_DRAWINGS_POS_CURRENT_MEAN"],
                     ccb.loc[:, "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"],
                     ccb.loc[:, "CC_CNT_DRAWINGS_POS_CURRENT_MEAN"],
                     ccb.loc[:, "CC_NEW_AMT_PER_ATM_DRAWING_MEAN"],
                     ccb.loc[:, "CC_NEW_AMT_PER_POS_DRAWING_MEAN"]], axis=1)

    return ccb


def prepare_instalment_payment():
    conn = pymysql.connect(host='35.228.28.142', port=int(63306), user='group2', passwd='123654', db='home_credit')
    df_installments_payments = pd.read_sql_query("SELECT * FROM installments_payments", conn)

    df_installments_payments = reduce_mem_usage(df_installments_payments)

    # O anki taksitin yuzde kaci odendi
    df_installments_payments[['AMT_PAYMENT']] = df_installments_payments[['AMT_PAYMENT']].fillna(value=0)
    df_installments_payments['NEW_INSTALMENT_PAYMENT_RATE'] = df_installments_payments['AMT_PAYMENT'] / \
                                                              df_installments_payments['AMT_INSTALMENT'] * 100

    # O anki taksit son odeme gununden kac gun once odenmis. Bu degisken "NEW_INSTALMENT_PAYMENT_STATUS" degerini bulabilmek icin gecici olusturulur.
    df_installments_payments['NEW_DAY_BEFORE_END_DATE'] = df_installments_payments['DAYS_INSTALMENT'] - \
                                                          df_installments_payments['DAYS_ENTRY_PAYMENT']

    df_installments_payments["NEW_INSTALMENT_PAYMENT_STATUS"] = "No Payment"
    df_installments_payments.loc[
        df_installments_payments['NEW_DAY_BEFORE_END_DATE'] == 0, "NEW_INSTALMENT_PAYMENT_STATUS"] = "In Time"
    df_installments_payments.loc[
        df_installments_payments['NEW_DAY_BEFORE_END_DATE'] > 0, "NEW_INSTALMENT_PAYMENT_STATUS"] = "Early"
    df_installments_payments.loc[
        df_installments_payments['NEW_DAY_BEFORE_END_DATE'] < 0, "NEW_INSTALMENT_PAYMENT_STATUS"] = "Late"

    df_installments_payments["NEW_INS_IS_LATE"] = "No"
    df_installments_payments.loc[df_installments_payments['NEW_DAY_BEFORE_END_DATE'] < 0, "NEW_INS_IS_LATE"] = "Yes"
    # Iki siniftan olustugu icin LabelEncoding yapilir.
    df_installments_payments = apply_label_encoding(df_installments_payments, ["NEW_INS_IS_LATE"])

    df_installments_payments.drop(columns=['NEW_DAY_BEFORE_END_DATE'], inplace=True)

    df_installments_payments, ip_cat = apply_one_hot_encoding(df_installments_payments)

    ip_aggregations = {
        'NUM_INSTALMENT_VERSION': ['max'],
        'NUM_INSTALMENT_NUMBER': ['max'],
        'AMT_INSTALMENT': ['sum'],
        'AMT_PAYMENT': ['sum'],
        'NEW_INSTALMENT_PAYMENT_RATE': ['min', 'max', 'mean'],
        'NEW_INS_IS_LATE': ['mean', 'sum']
    }

    for col in ip_cat:
        ip_aggregations[col] = ['mean']

    df_ip_agg = df_installments_payments.groupby(['SK_ID_CURR']).agg(ip_aggregations)

    df_ip_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in df_ip_agg.columns.tolist()])

    return df_ip_agg


def prepare_pos_cash_balance():
    conn = pymysql.connect(host='35.228.28.142', port=int(63306), user='group2', passwd='123654', db='home_credit')
    df_pos_cash_balance = pd.read_sql_query("SELECT * FROM POS_CASH_balance", conn)

    df_pos_cash_balance, pcb_cat = apply_one_hot_encoding(df_pos_cash_balance)

    pcb_aggregations = {
        'SK_ID_PREV': ['min', 'max', 'mean', 'count'],
        'MONTHS_BALANCE': ['min', 'max'],
        'CNT_INSTALMENT': ['min', 'max', 'mean'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }

    for col in pcb_cat:
        pcb_aggregations[col] = ['mean']

    df_pcb_agg = df_pos_cash_balance.groupby(['SK_ID_CURR']).agg(pcb_aggregations)
    df_pcb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in df_pcb_agg.columns.tolist()])

    return df_pcb_agg


def installment_payment_main():
    df_ip_agg = prepare_instalment_payment()  # Intaslment Payment son hali

    df_pcb_agg = prepare_pos_cash_balance()  # Pos cash balance son hali

    df_pos_ins = df_ip_agg.join(df_pcb_agg, how='inner',
                                on=['SK_ID_CURR'])  # instalment payment ve pos cash balance birlestirilmis hali

    return df_ip_agg, df_pcb_agg, df_pos_ins


def pre_processing_and_combine():
    with timer("Process application train"):
        df = application_train_g()
        print("application train & test shape:", df.shape)

    with timer("Bureau and Bureau Balance"):
        df_final = bureau_and_bureau_balance_features()
        print("Bureau and Bureau Balance:", df_final.shape)

    with timer("Installment Payments"):
        df_ip_agg, df_pcb_agg, df_pos_ins = installment_payment_main()
        print("Installment Payments", df_ip_agg.shape)

    with timer("Pos Cash Balance"):
        print("Pos Cash Balance:", df_pcb_agg.shape)

    with timer("Credit Card Balance"):
        ccb = credit_card_balance()
        print("Credit Card Balance:", ccb.shape)

    with timer("previous_application"):
        prev_agg = previous_application()
        print("previous_application:", prev_agg.shape)

    with timer("All tables are combining"):
        df = df.merge(df_final, how="left", on="SK_ID_CURR")
        df1 = df.merge(df_ip_agg, how='left', on='SK_ID_CURR')
        df2 = df1.merge(df_pcb_agg, how='left', on='SK_ID_CURR')
        df3 = df2.merge(ccb, how='left', on='SK_ID_CURR')
        all_df = df3.merge(prev_agg, how='left', on='SK_ID_CURR')

        print("all_df process:", all_df.shape)

    return all_df


def modeling(all_data):
    all_data = all_data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    train_df = all_data[all_data['TARGET'].notnull()]
    test_df = all_data[all_data['TARGET'].isnull()]

    folds = KFold(n_splits=10, shuffle=True, random_state=1001)

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()

    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]

        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

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

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=200, early_stopping_rounds=200)

        # y_pred_valid
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))  # y_pred_valid

    test_df['TARGET'] = sub_preds
    test_df[['SK_ID_CURR', 'TARGET']].to_csv("outputs/predictions/atilla_muhammet.csv'", index=False)

    display_importances(feature_importance_df)

    return feature_importance_df


def main():
    with timer("Preprocessing Time"):
        all_data = pre_processing_and_combine()

    with timer("Modeling"):
        feat_importance = modeling(all_data)


if __name__ == "__main__":
    with timer("Full model run"):
        main()
