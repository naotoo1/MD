FROM python:3.9.1
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
RUN pip install -r requirements.txt
CMD python malariadet_api.py