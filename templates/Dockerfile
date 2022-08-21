FROM python

RUN pip install pipenv && \
    python -m venv venv

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
ADD Pipfile* ./

RUN pipenv install --skip-lock

COPY app ./app
COPY assets ./assets

CMD pipenv run python -m app
