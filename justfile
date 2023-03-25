
lint:
    poetry run ruff check .
    poetry run mypy --strict .

format:
    poetry run black .

test:
    poetry run pytest tests
