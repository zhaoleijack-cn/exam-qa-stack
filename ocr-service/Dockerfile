FROM python:3.12-slim

ARG PIP_INDEX=https://pypi.org/simple
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -i ${PIP_INDEX} -r requirements.txt

COPY app.py ./

ENV PYTHONUNBUFFERED=1
EXPOSE 8090
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
