
lint:
    poetry run ruff check .
    poetry run mypy .

format:
    poetry run black .

test:
    poetry run pytest tests
