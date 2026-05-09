FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8502

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8502"]

