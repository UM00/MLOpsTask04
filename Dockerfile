
FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y make

RUN pip install Werkzeug==2.0.3

RUN make install

EXPOSE 5000

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

CMD ["flask", "run"]
