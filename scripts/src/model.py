import pandas as pd
from datetime import date, timedelta
import joblib
import logging
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

m_logger = logging.getLogger(__name__)
m_logger.setLevel(logging.DEBUG)
handler_m = logging.StreamHandler()
formatter_m = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
handler_m.setFormatter(formatter_m)
m_logger.addHandler(handler_m)

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

LAST_DATE = date(2023, 7, 18)  # last date from train data
FORECAST_STEP = 14  # number of forecast days


def collect(df: pd.DataFrame, raw_data: pd.DataFrame) -> pd.DataFrame:
    """reorganize dataframe from backend for model input, add features and lags"""
    extend_df = pd.DataFrame()
    df['date_today'] = pd.to_datetime(
        df['date_today'], format='%Y-%m-%d')

    num_days = 14
    data_list = [df['date_today'] + timedelta(days=x) for x in range(num_days)]
    column = df.index

    extend_df['date'] = data_list
    df_columns = ['st_id',
                  'pr_sku_id',
                  'pr_group_id',
                  'pr_cat_id',
                  'pr_subcat_id',
                  'pr_uom_id',
                  'st_city_id',
                  'st_division_code',
                  'st_type_format_id',
                  'st_type_loc_id',
                  'st_type_size_id']
    extend_df[df_columns] = df[column[1:12]]

    extend_df['day_of_week'] = extend_df['date'].dt.weekday
    extend_df['is_weekend'] = 0
    extend_df.loc[(extend_df['day_of_week'] == 5) | (extend_df['day_of_week'] == 6), 'is_weekend'] = 1

    extend_df['day_of_month'] = extend_df['date'].dt.day

    holiday = ['11.04.2022', '12.31.2022', '01.07.2023', '02.23.2023', '03.08.2023', '04.16.2023', '05.01.2023',
               '05.09.2023', '06.12.2023']
    extend_df['holiday'] = 0
    days = 3
    for i in holiday:
        for j in range(days + 1):
            extend_df.loc[extend_df['date'] == (pd.Timestamp(i) - pd.Timedelta(days=days - j)), 'holiday'] = (j + 1)

    lag = ['lag_21',
           'lag_20',
           'lag_19',
           'lag_18',
           'lag_17',
           'lag_16',
           'lag_15',
           'lag_14']
    raw_data['crutch'] = 0
    column = raw_data.columns
    for i in range(14):
        extend_df.loc[i, lag] = df[column[-9-i:-1-i]].values

    extend_df['rolling_mean'] = extend_df[lag[:-1]].mean(axis=1).values

    extend_df = extend_df[['date',
                           'st_id',
                           'pr_sku_id',
                           'pr_group_id',
                           'pr_cat_id',
                           'pr_subcat_id',
                           'pr_uom_id',
                           'st_city_id',
                           'st_division_code',
                           'st_type_format_id',
                           'st_type_loc_id',
                           'st_type_size_id',
                           'holiday',
                           'day_of_week',
                           'day_of_month',
                           'is_weekend',
                           'lag_14',
                           'lag_15',
                           'lag_16',
                           'lag_17',
                           'lag_18',
                           'lag_19',
                           'lag_20',
                           'lag_21',
                           'rolling_mean']]
    return extend_df


def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """preprocess data for forecast"""
    data = data.drop(['date'], axis=1)
    SCALER = joblib.load('scaler.save')
    ENCODER = joblib.load('encoder.save')
    data[CATEG] = ENCODER.transform(data[CATEG])
    data[NUMERICAL] = SCALER.transform(data[NUMERICAL])
    return data


def forecast(path: str) -> (list, str):
    """forecast function"""
    info = pd.read_csv(path)
    for col in info.columns:
        if 'lag' in col:
            info[col] = info[col].astype('float')
    info = info[100:200].reset_index(drop=True)  # change number of rows for tests
    info = info.loc[(info['store_id'] != "1aa057313c28fa4a40c5bc084b11d276") &
                    (info['store_id'] != "62f91ce9b820a491ee78c108636db089")]
    steps = int(info.shape[0])
    data_collected = pd.DataFrame()
    for j in range(len(info)):
        row = info.loc[j]
        data_collected = pd.concat([data_collected, collect(row, info)], axis=0)
    m_logger.info(f'data collected')
    f_dates = data_collected['date'].astype('str')
    query = preprocessing(data_collected)
    m_logger.info(f'data scaled and transformed')
    ESTIMATOR = joblib.load('rf.joblib')
    m_logger.info(f'model loaded')
    result = []
    status = 'OK'
    problem_pair_counter = 0
    for i in range(steps):
        start = i * FORECAST_STEP
        stop = (i + 1) * FORECAST_STEP
        forecast_dates = f_dates[start:stop]
        subquery = query[start:stop]
        subquery = subquery.dropna(axis=0)
        if len(subquery) == FORECAST_STEP:
            try:
                # predictions = np.random.randint(14, 88, len(subquery))  # for tests
                predictions = np.around(ESTIMATOR.predict(subquery), 0)
            except ValueError:
                m_logger.error(f'estimator fail')
                predictions = np.zeros(FORECAST_STEP, dtype=np.uint8)
                status = 'FAIL'
                pass
        else:
            predictions = np.zeros(FORECAST_STEP, dtype=np.uint8)
            m_logger.warning(f'for some store-product pairs there is no data to make forecast')
            problem_pair_counter += 1
        store = info.loc[i, 'store_id']
        item = info.loc[i, 'pr_sku_id']
        now_date = info.loc[i, 'date_today']
        result.append({"store": store,
                       "forecast_date": now_date,
                       "forecast": {"sku": item,
                                    "sales_units": {k: int(v) for k, v in zip(forecast_dates, predictions)}
                                    }
                       })
        i += 1
    m_logger.info(f'prediction finished')

    return result, status, problem_pair_counter
