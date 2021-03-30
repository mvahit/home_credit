import pandas as pd
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    import pandas as pd
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# command line access for debuging
def get_namespace():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=True)
    return parser.parse_args()


# i love data science
def i_love_ds():
    print('\n'.join([''.join([(' I_Love_Data_Science_'[(x - y) % len('I_Love_Data_Science_')]
                               if ((x * 0.05) ** 2 + (y * 0.1) ** 2 - 1) ** 3 - (x * 0.05) ** 2 * (
                y * 0.1) ** 3 <= 0 else ' ')
                              for x in range(-30, 30)]) for y in range(15, -15, -1)]))


# Display/plot feature importance
def display_importances(feature_importance_df_):
    import seaborn as sns
    import matplotlib.pyplot as plt
    cols = (feature_importance_df_[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:100].index)
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(10, 20))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('outputs/features/lgbm_importances.png')


# missing values
#
# def missing_values(df):
#
#     cols_with_na = [col for col in df.columns if df[col].isnull().sum() > 0]
#     for col in cols_with_na:
#         print(col, np.round(df[cols_with_na].isnull().mean(), 3), " % missing values")



# # saving models
# def saving_models():
#     import os
#     cur_dir = os.getcwd()
#     os.chdir('/models/reference/')
#     model_name = "lightgbm_fold_" + str(n_fold + 1) + "." + "pkl"
#     pickle.dump(model, open(model_name, 'wb'))  # model
#     os.chdir(cur_dir)

