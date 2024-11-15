FROM python:3.10-slim

WORKDIR /app

COPY programe/.env /app/programe/.env
ENV DOTENV_FILE=/app/.env


COPY programe /app/programe
COPY GUI /app/GUI


RUN pip install --no-cache-dir -r /app/programe/requirements.txt

EXPOSE 5000

CMD ["python", "/app/programe/lama_rag_JS.py"]
