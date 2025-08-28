#!/bin/bash

if [ ! -d venv ]; then
    python -m venv venv
    . venv/bin/activate
    pip install --upgrade pip
    pip install -e .
    pre-commit install
fi

source venv/bin/activate
