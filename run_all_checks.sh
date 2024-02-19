#!/bin/bash
set -e

poetry run black .
poetry run isort . --profile black
poetry run mypy .
poetry run pytest .