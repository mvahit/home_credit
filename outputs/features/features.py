# TODO feature isimlerini duzgunce al txt olarak bas


import pandas as pd
pd.set_option('display.max_columns', None)
df = pd.read_pickle("/Users/mvahit/Documents/GitHub/home_credit/outputs/features/feature_importance_df.pkl")
df = df.groupby("feature")["importance"].agg({"mean"}).sort_values(by="mean", ascending=False)


df[df["mean"] > 0]


df2 = pd.read_pickle("/Users/mvahit/Documents/GitHub/home_credit/outputs/features/fold_auc_best_df.pkl")

