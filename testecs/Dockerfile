FROM python:3.6-slim

COPY requirements.txt /app/
WORKDIR /app

RUN apt update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install --upgrade pip \
    &&  pip install --trusted-host pypi.python.org --requirement requirements.txt

COPY test.py /app

CMD ["python", "test.py"]
