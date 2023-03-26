
lint:
    poetry run ruff check .
    poetry run mypy --strict .

format:
    poetry run black .
    poetry run isort .

test:
    poetry run pytest tests
