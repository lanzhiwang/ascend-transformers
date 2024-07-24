# FROM docker-mirrors.alauda.cn/library/python:3.10.12-bullseye
FROM python:3.10.12-bullseye

RUN set -eux; \
    pip install transformers==4.37.1; \
    pip freeze
