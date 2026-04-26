import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, FunctionTransformer, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering, KMeans

from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN

seed = 420

def ttSplit(d, y_column, classes = False):

    if classes == True:

        x_tr, x_te, y_tr, y_te = train_test_split(
                d.drop(columns = [y_column]),
                d[y_column],
                test_size = .2,
                random_state = 420,
                stratify = d[y_column])

        return x_tr, x_te, y_tr, y_te

    else:
        
        x_tr, x_te, y_tr, y_te = train_test_split(
                d.drop(columns = [y_column]),
                d[y_column],
                test_size = .2,
                random_state = 420)

        return x_tr, x_te, y_tr, y_te

    return None, None, None, None

def forge(forger, x_tr, x_te):

    x_tr_tf = forger.fit_transform(x_tr)
    x_tr_tf = pd.DataFrame(x_tr_tf, index=x_tr.index, columns=x_tr.columns)
    
    x_te_tf = forger.transform(x_te)
    x_te_tf = pd.DataFrame(x_te_tf, index=x_te.index, columns=x_te.columns)
    
    return x_tr_tf, x_te_tf, forger

def wholeForge(forger, x_data):
    x_tf = forger.fit_transform(x_data)
    x_tf = pd.DataFrame(x_tf, index=x_data.index, columns=x_data.columns)

    return x_tf
    
def logTransform(x_tr, x_te):
    
    transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)

    return forge(transformer, x_tr, x_te)

def wholeLogTransform(x_data):
    
    transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)

    return wholeForge(transformer, x_data)

def boxCoxTransform(x_tr, x_te):

    transformer = pt = PowerTransformer(method='box-cox')
  
    return forge(transformer, x_tr, x_te)

def wholeBoxCoxTransform(x_data):

    transformer = pt = PowerTransformer(method='box-cox')
  
    return wholeForge(transformer, x_data)

def yjTransform(x_tr, x_te):

    transformer = pt = PowerTransformer(method='yeo-johnson', standardize=True)

    return forge(transformer, x_tr, x_te)
    
def standardScale(x_tr, x_te):
    
    scaler = StandardScaler()
    
    return forge(scaler, x_tr, x_te)

def minMaxScale(x_tr, x_te):
    
    scaler = MinMaxScaler()
    
    return forge(scaler, x_tr, x_te)

def robustScale(x_tr, x_te):
    
    scaler = RobustScaler()
    
    return forge(scaler, x_tr, x_te)

def imputeWithMeans(x_tr, x_te):

    imputer = SimpleImputer(strategy = 'mean')

    return forge(imputer, x_tr, x_te)

def simpleImputation(data):
    
    num_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    cat_cols = [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c])]
    
    # Impute numeric columns with median
    for col in num_cols:
        data[col] = data[col].fillna(data[col].median())
    
    # Impute categorical columns with mode
    for col in cat_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    return data


def randomUnderSampleClass(datas, yColumn, targetClass, factor = 2):

    count = 0
    
    for cl in np.unique(datas[yColumn]):

        if cl != targetClass:
            count += datas[datas[yColumn] == cl].shape[0]


    targetDf = datas[datas[yColumn] == targetClass]
    targetDf = targetDf.sample(n = int(count * factor), replace = False,
                               random_state = seed, axis = 0,
                               ignore_index = True)

    collateralDf = datas[datas[yColumn] != targetClass]

    df = pd.concat([collateralDf, targetDf], axis = 0, ignore_index = True)

    return df


def randomOverSample(x_data, y_data, sampling_strat = "auto"):
    
    le = LabelEncoder()
    y_data = le.fit_transform(y_data)

    strat = sampling_strat
    
    if isinstance(sampling_strat, dict):
        strat = dict()
        
        for key, value in sampling_strat.items():
            strat[le.transform([key])[0]] = value
    
    sampler = RandomOverSampler(sampling_strategy = strat)

    x_sm, y_sm = sampler.fit_resample(x_data, y_data)
    y_sm = le.inverse_transform(y_sm)

    x_sm = pd.DataFrame(x_sm, columns=x_data.columns)
    y_sm = pd.Series(y_sm)

    return x_sm, y_sm


def smoteSample(x_data, y_data, sampling_strat = "auto"):
    
    le = LabelEncoder()
    y_data = le.fit_transform(y_data)

    strat = sampling_strat
    
    if isinstance(sampling_strat, dict):
        strat = dict()
        
        for key, value in sampling_strat.items():
            strat[le.transform([key])[0]] = value
    
    sampler = SMOTE(sampling_strategy = strat)

    x_sm, y_sm = sampler.fit_resample(x_data, y_data)
    y_sm = le.inverse_transform(y_sm)

    x_sm = pd.DataFrame(x_sm, columns=x_data.columns)
    y_sm = pd.Series(y_sm)

    return x_sm, y_sm
    

def smoteennSample(x_data, y_data, sampling_strat = "auto"):
    
    le = LabelEncoder()
    y_data = le.fit_transform(y_data)

    strat = sampling_strat
    
    if isinstance(sampling_strat, dict):
        strat = dict()
        
        for key, value in sampling_strat.items():
            strat[le.transform([key])[0]] = value
    
    sampler = SMOTEENN(sampling_strategy = strat)

    x_sm, y_sm = sampler.fit_resample(x_data, y_data)
    y_sm = le.inverse_transform(y_sm)

    x_sm = pd.DataFrame(x_sm, columns=x_data.columns)
    y_sm = pd.Series(y_sm)

    return x_sm, y_sm

def tomekSample(x_data, y_data, sampling_strat = "auto"):
    
    le = LabelEncoder()
    y_data = le.fit_transform(y_data)

    strat = sampling_strat
    
    if isinstance(sampling_strat, list):
        strat = []
        
        for cl in sampling_strat:
            strat.append(le.transform([cl])[0])
    
    sampler = TomekLinks(sampling_strategy = strat)

    x_sm, y_sm = sampler.fit_resample(x_data, y_data)
    y_sm = le.inverse_transform(y_sm)

    return x_sm, y_sm

def cluster(data, columns, n, clusterer):
    
    data["cluster"] = clusterer.fit_predict(data[columns])

    return data

def aglCluster(data, columns, n):
    
    clusterer = AgglomerativeClustering(n_clusters = n, random_state = seed, linkage = "complete")

    return cluster(data, columns, n, clusterer)

def kmCluster(data, columns, n):
    
    clusterer = KMeans(n_clusters = n, random_state = seed, n_init = 10)

    return cluster(data, columns, n, clusterer)

def stringifyValue(value):
    
    return str(value)

def stringifyColumn(d, column):

    d[column] = d[column].apply(stringifyValue)

    return d