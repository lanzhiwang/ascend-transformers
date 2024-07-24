# FROM docker-mirrors.alauda.cn/library/python:3.10.12-bullseye
FROM python:3.10.12-bullseye

WORKDIR /work

COPY requirements.txt requirements.txt
COPY mlserver-example /work/mlserver-example

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates wget git git-lfs tree; \
    rm -rf /var/lib/apt/lists/*; \
    pip install -r requirements.txt; \
    pip freeze; \
    pwd; \
    tree -L 3 .
