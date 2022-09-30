SHELL := /bin/bash
python_version = 3.8.5

.PHONY: all test docs

lint:
	@echo -e "Running linter"
	@isort slu
	@black slu

install:
	@poetry install && pre-commit install

test:
	@pytest --cov=slu --cov-report html --cov-report term:skip-covered tests/
	@echo -e "The tests pass!"

build:
	@bash build.sh $1

all: lint install test build
