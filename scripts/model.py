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
FORECAST_STEP = 14  # длительность прогноза в будущее


def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """preprocess data for forecast.
    !!! ToDO:
    - add loading data from database and sales archive
    - add lags calculation
    - add holiday and weekend calculation
    - add cluster matrix loading and cluster calculation
    """
    data = data.drop(['date', 'forecast_date'], axis=1)
    try:
        data[CATEG] = ENCODER.transform(data[CATEG])
        m_logger.info(f'data transformed')
    except TypeError:
        m_logger.error(f'encoder problems')
        exit()
    data[NUMERICAL] = SCALER.transform(data[NUMERICAL])

    return data


def forecast(path: str) -> list:
    """forecast function"""
    info = pd.read_csv(path, index_col=0)
    steps = int(info.shape[0] / FORECAST_STEP)
    query = preprocessing(info)
    result = []
    for i in range(steps):
        start = i * FORECAST_STEP
        stop = (i + 1) * FORECAST_STEP
        subquery = query[start:stop]
        forecast_dates = info.loc[start:stop, 'forecast_date']
        predictions = np.around(ESTIMATOR.predict(subquery), 0)
        store = info.loc[start, 'st_id']
        item = info.loc[start, 'pr_sku_id']
        now_date = info.loc[start, 'date']
        result.append({"store": store,
                       "forecast_date": now_date,
                       "forecast": {"sku": item,
                                    "sales_units": {k: v for k, v in zip(forecast_dates, predictions)}
                                    }
                       })
        i += 1
    return result
