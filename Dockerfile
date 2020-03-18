FROM python:3.7

WORKDIR '/deploy'

COPY ./requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "predict.py"]