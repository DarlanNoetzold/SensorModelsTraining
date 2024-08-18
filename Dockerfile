FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Configuração do cron job para rodar diariamente às 02:00
RUN apt-get update && apt-get install -y cron && \
    echo "0 2 * * * root python /app/train_models.py > /proc/1/fd/1 2>/proc/1/fd/2" >> /etc/crontab

CMD ["cron", "-f"]
