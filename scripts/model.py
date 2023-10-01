import pandas as pd
from datetime import date, timedelta
import joblib
import logging
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

m_logger = logging.getLogger(__name__)
m_logger.setLevel(logging.DEBUG)
handler_m = logging.StreamHandler()
formatter_m = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
handler_m.setFormatter(formatter_m)
m_logger.addHandler(handler_m)

SCALER = joblib.load('scaler.save')

ENCODER = joblib.load('encoder.save')

ESTIMATOR = joblib.load('rf.joblib')
m_logger.info(f'model loaded')

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
    data = data.drop(['date', 'Unnamed: 0'], axis=1)
    try:
        data[CATEG] = ENCODER.transform(data[CATEG])
        m_logger.info(f'data transformed')
    except TypeError:
        m_logger.error(f'encoder problems')
        exit()
    data[NUMERICAL] = SCALER.transform(data[NUMERICAL])

    return data


def forecast(path: str) -> list:
    """forecast function
    !!! ToDO:
    - correct output format
    """
    info = pd.read_csv(path)
    query = preprocessing(info)
    predictions = np.around(ESTIMATOR.predict(query), 0)
    result = []
    for i in range(len(predictions)):
        store = info.loc[i, 'st_id']
        item = info.loc[i, 'pr_sku_id']
        # forecast_dates = [LAST_DATE + timedelta(days=d) for d in range(1, 14)]
        # forecast_dates = [el.strftime("%Y-%m-%d") for el in forecast_dates]
        f_date = info.loc[i, 'date']
        result.append({"store": store,
                       "forecast_date": f_date,
                       "forecast": {"sku": item,
                                    "sales_units": int(predictions[i])}
                      })
    return result
