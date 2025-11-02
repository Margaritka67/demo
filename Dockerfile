FROM python:3.12-slim

WORKDIR /app
COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

ENV CHAT_MODEL="gpt-4o-mini"
ENV CHAT_API_BASE="https://api.openai.com/v1"
ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]

