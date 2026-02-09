import numpy as np 
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipelines(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
    ("impute" , SimpleImputer(strategy="median")),
    ("standardize" , StandardScaler()),
    ])
    cat_pipeline = Pipeline([
    ('onehot' , OneHotEncoder()),
    ])
    full_pipeline = ColumnTransformer([
    ('num' , num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs),
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv('housing.csv')

    df['income_cat'] = pd.cut(df['median_income'], bins = [0, 1.5, 3, 4.5, 6, np.inf], labels = [1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index , test_index in split.split(df, df['income_cat']):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
        
    for remove_columns in (strat_test_set, strat_train_set):
        remove_columns.drop('income_cat', axis = 1, inplace = True)

    housing = strat_train_set.copy()

    housing_label = housing['median_house_value'].copy()
    housing = housing.drop('median_house_value', axis = 1)

    num_attribs = housing.drop('ocean_proximity', axis = 1).columns.tolist()
    cat_attribs = ['ocean_proximity']

    pipeline = build_pipelines(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing)
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_label)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv')
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions

    input_data.to_csv("output.csv", index=False)
    print("Inference is complete, results saved to output.csv Enjoy!")
