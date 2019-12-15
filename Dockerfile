FROM intrepidlemon/cuda-python:latest

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update \
    && apt-get install -y git

COPY requirements.txt /tmp/

RUN python3.6 -m pip install -r /tmp/requirements.txt

WORKDIR /usr/src/app

COPY . .

ENV DATA_DIR=/data
