FROM vernacularai/ubuntu-pyenv:1.0.0


ARG PYTHON_VER=3.6.9
ARG POETRY_VER=1.0.2
ARG VERSION

WORKDIR /home/slu

RUN pyenv install $PYTHON_VER && pyenv global $PYTHON_VER && \ 
    pip install poetry==$POETRY_VER && \ 
    apt-get update && apt-get install -y gcc g++ && \ 
    pip install --upgrade pip

RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock /home/slu/

RUN poetry install --no-dev && \ 
    rm -rf /root/.cache && pip3 uninstall --yes poetry

COPY . .

ENV GOOGLE_API_KEY="" \
    DUCKLING_URL="" \
    ENVIRONMENT="PRODUCTION" \
    GOOGLE_API_KEY="" \
    SENTRY_DSN=""


EXPOSE 8005

CMD ["/bin/sh", "-ec", "while :; do echo '.'; sleep 5 ; done"]
