SHELL := /bin/bash
python_version = 3.8.5

.PHONY: all test docs

lint:
	@echo -e "Running linter"
	@isort slu
	@black slu

test:
	@pytest --cov=slu --cov-report html --cov-report term:skip-covered tests/
	@echo -e "The tests pass!"

build:
	@bash build.sh $1

all: lint test build
