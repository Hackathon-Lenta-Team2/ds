import argparse
import datetime
import logging
import json
import uvicorn
from fastapi import FastAPI
from model import forecast
import requests

app = FastAPI()

dataDir = 'tmp/'

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.DEBUG)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)


@app.get("/ready")
def forecast_ready():
    pass


@app.get("/")
def main(path='test_missing.csv') -> tuple:  # здесь должно быть нормальное получение файла / пути к файлу
    """runs forecast and save result"""
    app_logger.info(f'data successfully loaded')
    result, status = forecast(path)
    message = 'forecast successfully finished, results saved'
    if status != 'OK':
        app_logger.error(f'forecast failed')
        message = 'forecast failed'
    else:
        app_logger.info(f'forecast finished')
        with open(dataDir + 'forecast_archive.json', 'w') as file:
            json.dump(result, file)
            app_logger.info(f'data saved')
    app_logger.info(message)
    resp = requests.get("http://localhost:8001/ready")  # пока что запрос просто по какому-то адресу
    result = {'date': datetime.datetime.now(), 'status': message}
    if resp.status_code != 200:
        result = 'something wrong'
    return result, resp.status_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8001, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())
    uvicorn.run(app, **args)
