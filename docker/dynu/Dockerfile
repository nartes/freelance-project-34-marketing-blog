FROM python:3
WORKDIR /app

RUN pip3 install requests

ENTRYPOINT exec python3 d1/dynu_update tmp/cache/dynu.auth.json ngrok
