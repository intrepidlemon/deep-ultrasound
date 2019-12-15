FROM intrepidlemon/cuda-python:latest

COPY requirements.txt /tmp/

RUN python3.6 -m pip install -r /tmp/requirements.txt

WORKDIR /usr/src/app

COPY . .

ENV DATA_DIR=/data
