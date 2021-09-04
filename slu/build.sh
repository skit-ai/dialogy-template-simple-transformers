#!/bin/bash

APP_NAME=$1

models=$(echo data/* | wc | awk '{print $2}')
if (( $models > 1 )); then
    echo "More than one model found in data/ directory. Please remove all but one."
    exit 1
fi

docker build -t $APP_NAME .
