FROM python:3.11-slim

# نصب ffmpeg برای پردازش صدا و تصویر
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir httpx yt-dlp fastapi uvicorn

COPY . .

CMD ["python", "main.py"]