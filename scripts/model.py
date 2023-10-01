def forecast(sales: dict, item_info: dict, store_info: dict) -> list:
    """
    Функция для предсказания продажЖ
    :params sales: исторические данные по продажам
    :params item_info: характеристики товара
    :params store_info: характеристики магазина

    """
    sales = [el["sales_units"] for el in sales]
    mean_sale = sum(sales) / len(sales)
    return [mean_sale] * 5

features = df.drop('target', axis=1)
target = df.target

train_features, test_features, train_target, test_target = train_test_split(
    features,
    target,
    test_size=.25,
    shuffle=False
    )
numerical = [
    'holiday',
    'rolling_mean',
    'lag_14',
    'lag_15',
    'lag_16',
    'lag_17',
    'lag_18',
    'lag_19',
    'lag_20',
    'lag_21'
    ]
scaler = RobustScaler()
train_features[numerical] = scaler.fit_transform(train_features[numerical])
test_features[numerical] = scaler.transform(test_features[numerical])
tss = TimeSeriesSplit(n_splits=8)
Код обучения модельки:

rf_params = {
    'n_estimators': list(range(100, 201)),
    'max_depth': list(range(20, 51)),
    'min_samples_leaf': list(range(5, 21)),
    'max_features': [.3, .5, .7, 1, 'sqrt', 'log2']
    }
rf_model = RandomForestRegressor(random_state=SEED, criterion='squared_error')
best_rf_model = TuneSearchCV(
    rf_model,
    rf_params,
    scoring='neg_root_mean_squared_error',
    cv=tss,
    n_trials=100,
    n_jobs=-1,
    random_state=SEED,
    search_optimization='hyperopt',
    verbose=1
    )
best_rf_model.fit(train_features, train_target)

import pickle
from itertools import combinations

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.figure_factory import create_dendrogram

import ydata_profiling as yp
from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.tsa.seasonal import seasonal_decompose

from tune_sklearn import TuneSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

Это только для того, что выше кинул, по идее:

import pickle

import pandas as pd

from tune_sklearn import TuneSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit

SEED=333