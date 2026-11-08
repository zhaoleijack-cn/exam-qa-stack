FROM python:3.12-slim-bookworm

ARG PIP_INDEX=https://pypi.org/simple
WORKDIR /data

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -i ${PIP_INDEX} -r /tmp/requirements.txt

COPY *.py /app-src/
COPY index.html /app-src/index.html

ENV PYTHONUNBUFFERED=1
EXPOSE 8090
# index.html 按约定放在数据卷 /data；首次启动若卷里没有则自动播种
CMD ["sh", "-c", "[ -f /data/index.html ] || cp /app-src/index.html /data/; exec uvicorn main:app --host 0.0.0.0 --port 8090 --app-dir /app-src"]
