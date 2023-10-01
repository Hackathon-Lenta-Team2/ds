import pandas as pd
import pickle
from datetime import date, timedelta

with open('scaler.pkl', 'rb') as f_1:
    SCALER = pickle.load(f_1)

with open('encoder.pkl', 'rb') as f_2:
    ENCODER = pickle.load(f_2)

with open('rf.pkl', 'rb') as f_3:
    ESTIMATOR = pickle.load(f_3)

NUMERICAL = [
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

CATEG = [
    'st_id',
    'pr_sku_id',
    'pr_group_id',
    'pr_cat_id',
    'pr_subcat_id',
    'st_city_id',
    'st_division_code',
    'st_type_format_id',
    'st_type_loc_id',
    'st_type_size_id'
    ]


LAST_DATE = date(2023, 7, 18)  # последняя дата из train


def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """preprocess data for forecast.
    !!! ToDO:
    - add loading data from database and sales archive
    - add lags calculation
    - add holiday and weekend calculation
    - add cluster matrix loading and cluster calculation
    """
    data = data.drop(['date'], axis=1)
    data[CATEG] = ENCODER.transform(data[CATEG])
    data[NUMERICAL] = SCALER.transform(data[NUMERICAL])

    return data


def forecast(path: str) -> list:
    """forecast function
    !!! ToDO:
    - correct output format
    """
    info = pd.read_csv(path)
    query = preprocessing(info)
    result = []
    for i in range(len(query)):
        row = query.loc[i, :]
        store = info.loc[i, 'st_id']
        item = info.loc[i, 'pr_sku_id']
        # forecast_dates = [LAST_DATE + timedelta(days=d) for d in range(1, 14)]
        # forecast_dates = [el.strftime("%Y-%m-%d") for el in forecast_dates]
        f_date = info.loc[i, 'date']
        prediction = ESTIMATOR.predict(row)
        result.append({"store": store,
                       "forecast_date": f_date,
                       "forecast": {"sku": item,
                                    "sales_units": prediction}
                      })
    return result
