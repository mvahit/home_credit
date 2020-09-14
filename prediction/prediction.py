import os
import pickle
import pandas as pd

os.chdir('/Users/mvahit/Documents/GitHub/home_credit/models/reference')

# files = os.listdir("/Users/mvahit/Documents/GitHub/home_credit/models/reference")

model = pickle.load(open('lightgbm_fold_6.pkl', 'rb'))

df = pd.read_pickle("/Users/mvahit/Documents/GitHub/home_credit/data/final_test_df.pkl")

model.predict(df)

# models = []

# for i in files:
    #models.append(pickle.load(open(i, 'rb')))

# model = pickle.load(open('regression_model.pkl', 'rb'))