# TODO feature isimlerini duzgunce al txt olarak bas


import pandas as pd
pd.set_option('display.max_columns', None)
df = pd.read_pickle("/Users/mvahit/Documents/GitHub/home_credit/outputs/features/feature_importance_df.pkl")
df.head()
df = df.groupby("feature")["importance"].agg({"mean"}).sort_values(by="mean", ascending=False)
df.head()

df[df["mean"] > 0]
df.shape

df2 = pd.read_pickle("/Users/mvahit/Documents/GitHub/home_credit/outputs/features/fold_auc_best_df.pkl")


# FINAL DF

import pandas as pd
pd.set_option('display.max_columns', None)
df = pd.read_pickle("/Users/mvahit/Documents/GitHub/home_credit/data/final_train_df.pkl")
df.head()
df.shape

[col for col in df.columns if col.startswith("APP")]
a = df[[col for col in df.columns if col.startswith("APP")]].head()