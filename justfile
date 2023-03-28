
lint:
    poetry run ruff check .
    poetry run mypy --strict .

format:
    poetry run black .
    poetry run isort .

test:
    poetry run pytest tests

test-dbg:
    poetry run pytest --pdb tests

audit:
    poetry export -f requirements.txt --output requirements.txt
    pip-audit -r requirements.txt

coverage:
    poetry run pytest --cov=excelsolver tests/

ci:
    tox -e py310
