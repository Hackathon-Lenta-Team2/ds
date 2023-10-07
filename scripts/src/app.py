import argparse
import datetime
import logging
import json
import uvicorn
from fastapi import FastAPI, BackgroundTasks
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


def make_forecast(path: str) -> tuple:
    """runs forecast and save result"""
    app_logger.info(f'data successfully loaded')
    result, status, problem_pairs = forecast(path)
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
    # resp = requests.get("http://localhost:8001/ds/ready")  # for local tests
    resp = requests.get("http://localhost/api/v1/import-data/")
    result = {'date': datetime.datetime.now(), 'status': message, 'problem pairs number': problem_pairs}
    if resp.status_code != 200:
        result = 'something wrong'
    print(result)
    print(resp.status_code)
    return result, resp.status_code


# for local tests
# @app.get("/ds/ready")
# def forecast_ready():
#     pass


@app.get("/ds/start")
async def main(background_tasks: BackgroundTasks) -> dict:
    """running forecast in background"""
    background_tasks.add_task(make_forecast, path=dataDir + 'ds_data.csv')
    return {"message": "forecast is running. wait please"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())
    uvicorn.run(app, **args)
