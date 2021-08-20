FROM python:3.6-slim

COPY requirements.txt /app/requirements.txt
WORKDIR /app

RUN apt update
RUN mkdir certificates
RUN mkdir checkpoint_variations_5_04
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install --upgrade pip \
    &&  pip install --trusted-host pypi.python.org --requirement requirements.txt

COPY ./ /app/

EXPOSE 80
CMD ["python", "app.py"]
