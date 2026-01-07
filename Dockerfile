# FROM python:3.10-slim-buster

# WORKDIR /app

# COPY . /app

# RUN pip install -r requirements.txt

# CMD ["python3", "app.py"]


FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8080

CMD ["python", "app.py"]
