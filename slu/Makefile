SHELL := /bin/bash
python_version = 3.8.5

lint:
	@echo -e "Running linter"
	@isort slu
	@black slu

test: lint ## Run the tests.
	@pytest --cov=src --cov-report html --cov-report term:skip-covered tests/
	@echo -e "The tests pass!" 

all: test
