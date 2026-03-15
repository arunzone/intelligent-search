FROM python:3.12-slim

WORKDIR /app

COPY setup.cfg .
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "intelligent_search"]
