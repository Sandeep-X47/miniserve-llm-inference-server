FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY benchmark.py .

EXPOSE 8000
ENV ENGINE=mock
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
