import requests
import os
import logging

from model import forecast

URL_CATEGORIES = "categories"
URL_SALES = "sales"
URL_STORES = "shops"
URL_FORECAST = "forecast"

api_port = os.environ.get("API_PORT", "8000")
api_host = os.environ.get("API_PORT", "localhost")

_logger = logging.getLogger(__name__)


def setup_logging():
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.DEBUG)
    handler_m = logging.StreamHandler()
    formatter_m = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
    handler_m.setFormatter(formatter_m)
    _logger.addHandler(handler_m)


def get_address(resource):
    return "http://" + api_host + ":" + api_port + "/" + resource


def get_stores():
    stores_url = get_address(URL_STORES)
    resp = requests.get(stores_url)
    if resp.status_code != 200:
        _logger.warning("Could not get stores list")
        return []
    return resp.json()["data"]


def get_sales(store=None, sku=None):
    sale_url = get_address(URL_SALES)
    params = {}
    if store is not None:
        params["store"] = store
    if sku is not None:
        params["sku"] = sku
    resp = requests.get(sale_url, params=params)
    if resp.status_code != 200:
        _logger.warning("Could not get sales history")
        return []
    return resp.json()["data"]


def get_categs_info():
    categs_url = get_address(URL_CATEGORIES)
    resp = requests.get(categs_url)
    if resp.status_code != 200:
        _logger.warning("Could not get category info")
        return {}
    result = {el["sku"]: el for el in resp.json()["data"]}
    return result


# Функции для работы с БД пока что не используются по причине отсутствия БД
# Загрузка данных идет просто из тестового файла

PATH = 'example.csv'  # для теста. имитация загрузки данных из БД по магазинам и товарам и архива продаж


def main(path: str):
    result = forecast(PATH)
    # requests.post(get_address(URL_FORECAST), json={"data": result})
    print(result)


if __name__ == "__main__":
    setup_logging()
    main(PATH)