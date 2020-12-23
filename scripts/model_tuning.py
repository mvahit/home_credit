"""Model tuning scripti calistiginda hyperparameters klasörüne iki sonuc uretecek:

    hyperparameters.pkl
    lightgbm_model.pkl

"""

# TODO feature isimlerini modele sokarak bu feature'lar ile tuning

import os
import pickle
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV

lgbm = LGBMClassifier()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [200, 100]}

df = pd.read_pickle("data/final_train_df.pkl")


y_train = df["TARGET"]

X_train = df.drop("TARGET", axis=1)

lgbm_cv_model = GridSearchCV(lgbm,
                             lgbm_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

dir(lgbm_cv_model)
params = lgbm_cv_model.best_params_

# saving hyperparameters and model
cur_dir = os.getcwd()
os.chdir('outputs/hyperparameters/')
pickle.dump(params, open("hyperparameters.pkl", 'wb'))  # hyperparameters
pickle.dump(lgbm_cv_model, open("lightgbm_model.pkl", 'wb'))  # model
os.chdir(cur_dir)

print("Best hyperparameters", params)


# loading and prediction with model

# del lgbm_cv_model
cur_dir = os.getcwd()
os.chdir('/Users/mvahit/Documents/GitHub/home_credit/outputs/hyperparameters/')
model = pickle.load(open('lightgbm_model.pkl', 'rb'))
os.chdir(cur_dir)
model.predict(X_train.head())

# loading hyperparameters
del model
del params
cur_dir = os.getcwd()
os.chdir('/Users/mvahit/Documents/GitHub/home_credit/outputs/hyperparameters/')
params = pickle.load(open('hyperparameters.pkl', 'rb'))
final_lgbm = LGBMClassifier(**params).fit(X_train, y_train)
final_lgbm.get_params()
final_lgbm.predict(X_train.head())

