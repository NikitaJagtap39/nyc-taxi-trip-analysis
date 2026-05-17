
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 100 -r requirements.txt

COPY predict.py .
COPY models/ models/

EXPOSE 9696

CMD ["python", "predict.py"]
