FROM python:3
WORKDIR /app

RUN pip3 install requests

ENTRYPOINT /bin/sh -c "\
  python3 d1/dynu_update.py tmp/cache/dynu.auth.json ngrok \
"

