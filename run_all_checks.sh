#!/bin/bash
set -e

poetry run black mash
poetry run isort mash --profile black
poetry run mypy mash
poetry run pytest mash
