
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def row_to_column(row):
    if type(row) == type(pd.Series()):
        return pd.DataFrame(row)
    elif type(row) == type(np.empty(0)):
        return row.reshape(-1,1)
    else:
        raise ValueError("row_to_column expected pandas Series or 1D numpy array")
        
class Row_Flipper(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        return self
    
    def row_to_column(row):
        if type(row) == type(pd.Series()):
            return pd.DataFrame(row)
        elif type(row) == type(np.empty(0)):
            return row.reshape(-1,1)
        else:
            raise ValueError("row_to_column expected pandas Series or a numpy array")
        
    def transform(self, X):
        if len(X.shape) != 1:
            raise ValueError('RowFlipper expected 1D object, but got argument of shape {X.shape}')
        return row_to_column(X)


class Discretizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, name, tresholds, handle_missing=True):
        self.name = name
        self.tresholds = tresholds
        self.handle_missing = handle_missing
        self.columns = []
        self.inf_tresholds = [-np.inf] + tresholds + [np.inf]
        
        for index in range(len(self.inf_tresholds) - 1):
            self.columns.append(f"{self.name}_({self.inf_tresholds[index]},{self.inf_tresholds[index+1]}]")
        if self.handle_missing:
            self.columns.append(f"{self.name}_missing")
        return None
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        one_hots = np.zeros( (len(self.inf_tresholds) - 1 + int(self.handle_missing), len(X)) )
        for index in range(len(self.inf_tresholds) - 1):
            one_hots[index] = (X > self.inf_tresholds[index]) & (X <= self.inf_tresholds[index + 1])
            
        if self.handle_missing:
            one_hots[len(self.inf_tresholds) - 1] = X.isnull()
        elif X.isnull().any():
            raise ValueError("There's NAN's in the input")

        result = pd.DataFrame(one_hots.T, columns=self.columns, index = X.index)   
        return result

    

FareTransformer = Pipeline([('lof1p', make_pipeline(Row_Flipper(), SimpleImputer(strategy='median'), FunctionTransformer(np.log1p, validate=False))),
                            ('scaler', MinMaxScaler(feature_range=(0, 1)))
                           ])

def numerical_transformer():
    numerical_tr = []
    numerical_tr.append(('age_discretizer', Discretizer('age', tresholds = [5,10,20,30,40,50]), 'Age'))
    numerical_tr.append(('sibsp_discretizer', Discretizer('sibsp', tresholds = [0, 1], handle_missing=False), 'SibSp'))
    numerical_tr.append(('parch_discretizer', Discretizer('parch', tresholds = [0], handle_missing=False), 'Parch'))
    numerical_tr.append(('fare_transformer', FareTransformer, 'Fare'))
    return ColumnTransformer(numerical_tr, sparse_threshold=0)


sex_encoder = make_pipeline(Row_Flipper(), OrdinalEncoder())


def take_num_len56(ticket):
    try:
        num_str = re.match('(\d+)$', ticket).group(1)
        return len(num_str) if len(num_str) in [5, 6] else -1
    except:
        return 0

def take_first_digit123(ticket):
    try:
        num_str = re.match('(\d+)$', ticket).group(1)
        return int(num_str[0]) if int(num_str[0]) in [1,2,3] else -1
    except:
        return 0
    
ticket_transformer_1 = make_pipeline(FunctionTransformer(np.vectorize(take_num_len56), validate=False), Row_Flipper(), OneHotEncoder(categories='auto'))
ticket_transformer_2 = make_pipeline(FunctionTransformer(np.vectorize(take_first_digit123), validate=False), Row_Flipper(), OneHotEncoder(categories='auto'))


def quasilist_to_set(quasilist):
    try:
        return frozenset(quasilist)
    except:
        return quasilist

def extract_letter(X):
    PATTERN = '([A-Z])\d*'
    result = X.str.findall(PATTERN)
    result[result.isnull()] = np.nan
    return result.apply(quasilist_to_set)

letter_extractor = FunctionTransformer(extract_letter, validate=False)

letters = 'ABCDE'
letter_to_num = {frozenset(letter): number for (letter, number) in zip(letters, range(1,len(letters)+1))}
letter_to_num[np.nan] = 0

def letter_categories(quasilist):
    return letter_to_num[quasilist] if quasilist in letter_to_num else -1

letter_encoder = FunctionTransformer(np.vectorize(letter_categories), validate=False)

cabin_transformer = make_pipeline(letter_extractor, letter_encoder, Row_Flipper(), OneHotEncoder(categories='auto'))


embarked_transformer = make_pipeline(Row_Flipper(), SimpleImputer(strategy='most_frequent'), OneHotEncoder(categories='auto'))


pclass_transformer = make_pipeline(Row_Flipper(), OneHotEncoder(categories='auto', handle_unknown='ignore'))


categorical_tr = []
categorical_tr.append(('sex_encoder', sex_encoder, 'Sex'))
categorical_tr.append(('ticket_transformer_1', ticket_transformer_1, 'Ticket'))
categorical_tr.append(('ticket_transformer_2', ticket_transformer_2, 'Ticket'))
categorical_tr.append(('cabin_transformer', cabin_transformer, 'Cabin'))
categorical_tr.append(('embarked_transformer', embarked_transformer, 'Embarked'))
categorical_tr.append(('pclass_transformer', pclass_transformer, 'Pclass'))
categorical_transformer = ColumnTransformer(categorical_tr, sparse_threshold=0)


def preprocessing_pipe():
    transformers = [
        ('numerical_transformer', numerical_transformer()),
        ('categorical_transformer', categorical_transformer)
    ]
    return FeatureUnion(transformers)
