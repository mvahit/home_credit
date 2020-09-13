import os
import pickle


os.chdir('/Users/mvahit/Documents/GitHub/home_credit/models/reference')

files = os.listdir("/Users/mvahit/Documents/GitHub/home_credit/models/reference")

model = pickle.load(open('lightgbm_fold_6.pkl', 'rb'))

#models = []

#for i in files:
    #models.append(pickle.load(open(i, 'rb')))

#model = pickle.load(open('regression_model.pkl', 'rb'))