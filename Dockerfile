FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir schedule webull-python-sdk-core webull-python-sdk-quotes-core webull-python-sdk-mdata webull-python-sdk-trade-events-core webull-python-sdk-trade

COPY . .

CMD ["python", "-u", "master_scheduler.py"]
