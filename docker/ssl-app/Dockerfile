FROM alpine:latest
RUN apk add openssh
RUN apk add python3
RUN apk add bash curl
RUN apk add py3-pip
RUN apk add nginx
RUN apk add tini
#RUN pip3 install requests certbot
RUN apk add certbot

WORKDIR /app

ENTRYPOINT /bin/sh -c "\
  python3 d1/nginx_config.py ssl tmp/d1/ssl.nginx.json /etc/nginx/nginx.conf && \
  tini -- nginx -g 'daemon off;' \
"
