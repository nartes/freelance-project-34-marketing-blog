FROM alpine:latest
RUN apk add openssh
RUN apk add python3
RUN apk add tini
RUN apk add bash curl
RUN apk add py3-pip
RUN pip3 install requests

WORKDIR /app

ENTRYPOINT ["tini", "--"]

CMD ["python3", "d1/cpanel.py"]
