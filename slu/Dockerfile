FROM 536612919621.dkr.ecr.ap-south-1.amazonaws.com/ubuntu-pyenv:latest


ARG PYTHON_VER=3.8.2
ARG POETRY_VER=1.1.13

WORKDIR /home/slu

RUN apt-get update && apt-get install -y gcc g++ && \
    pyenv install $PYTHON_VER && pyenv global $PYTHON_VER && \
    pip install --upgrade pip && \
    pip install poetry==$POETRY_VER

RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock data.dvc /home/slu/

RUN poetry install --no-dev && \
    rm -rf /root/.cache && pip3 uninstall --yes poetry && pip3 install -U setuptools

COPY ./.dvc ./.dvc

RUN dvc config core.no_scm true
RUN dvc pull

COPY ./slu ./slu
COPY ./config ./config
COPY ./server_fault_notif.py ./server_fault_notif.py

ENV GOOGLE_MAPS_API_TOKEN=$GOOGLE_MAPS_API_TOKEN
ENV MMI_CLIENT_ID=$MMI_CLIENT_ID
ENV MMI_CLIENT_SECRET=$MMI_CLIENT_SECRET

EXPOSE 8005

CMD ["/bin/sh", "-ec", "while :; do echo '.'; sleep 5 ; done"]
