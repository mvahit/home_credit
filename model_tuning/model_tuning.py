from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV

lgbm = LGBMClassifier()

lgbm_params = {"learning_rate": [0.001, 0.01, 0.1],
               "n_estimators": [200, 500, 100]}

df = pd.read_pickle("/Users/mvahit/Documents/GitHub/home_credit/data/final_train_df.pkl")


y_train = df["TARGET"]

X_train = df.drop("TARGET", axis=1)

lgbm_cv_model = GridSearchCV(lgbm,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)


print(lgbm_cv_model.best_params_)

lgbm_cv_model.best_params_.to_pickle("/Users/mvahit/Documents/GitHub/home_credit/model_tuning/hiperparameters.pkl")


