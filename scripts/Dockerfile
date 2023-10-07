FROM python:3.10

COPY requirements.txt /scripts/requirements.txt
COPY src /scripts
WORKDIR /scripts

RUN apt-get update && apt-get install
RUN pip install -r requirements.txt

CMD ["python", "app.py"]

