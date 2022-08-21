dev:
	pipenv run python -m app

install:
	pipenv install

test:
	pipenv run pytest -vvv -n auto --dist load
