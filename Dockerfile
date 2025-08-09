FROM python:3.11-slim
RUN apt-get update && apt-get install -y libgl1
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
