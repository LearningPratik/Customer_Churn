import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder
)

# data_path = 'D:/Projects/churn_project/data/churn_encoded.csv'

def split_data(data_path):

    df = pd.read_csv(data_path)
    df = df.drop(columns='Unnamed: 0')
    X = df.drop(columns='churn', axis = 1)
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 1)

    return X_train, X_test, y_train, y_test

def Ohe():

    ohe_preprocessor = ColumnTransformer(
        transformers = [

            ('cat_2', OneHotEncoder(drop = 'first', sparse_output=False), ['gender', 'partner', 'dependents', 'phoneservice', 'paperlessbilling']),
            ('cat_3', OneHotEncoder(drop = 'first', sparse_output=False), ['multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies']),
            ('cat_4', OneHotEncoder(drop = 'first', sparse_output=False), ['contract', 'paymentmethod']),
            ('charges', StandardScaler(), ['monthlycharges', 'totalcharges']),
            
        ], remainder = 'passthrough', force_int_remainder_cols = False
        
    )

    return ohe_preprocessor