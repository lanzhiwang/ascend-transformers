# FROM docker-mirrors.alauda.cn/library/python:3.10.12-bullseye
FROM python:3.10.12-bullseye

COPY requirements.txt requirements.txt
RUN set -eux; \
    pip install -r requirements.txt; \
    pip freeze
