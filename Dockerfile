# FROM docker-mirrors.alauda.cn/library/python:3.10.12-bullseye
FROM python:3.10.12-bullseye

WORKDIR /work

COPY requirements.txt requirements.txt
COPY mlserver-example /work/mlserver-example

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates wget git git-lfs tree libhdf5-dev; \
    rm -rf /var/lib/apt/lists/*; \
    pip install -r requirements.txt; \
    pip freeze; \
    pwd; \
    tree -L 3 .; \
    cd mlserver-example/02-Serving-HuggingFace-models; \
    git lfs install; \
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/openai-community/gpt2; \
    tree -L 5 /work/mlserver-example; \
    pip cache purge
