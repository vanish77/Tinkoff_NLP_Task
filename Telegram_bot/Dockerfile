FROM python:latest

COPY . /tg_bot

WORKDIR /tg_bot

RUN pip3 install -r requirements.txt

CMD [ "python3", "main.py" ]