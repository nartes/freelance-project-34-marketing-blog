FROM nginx:latest
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3
WORKDIR /app

ENTRYPOINT /bin/sh -c "\
  python3 d1/nginx_config.py tmp/cache/forward.nginx.json /etc/nginx/nginx.conf && \
  /docker-entrypoint.sh nginx -g 'daemon off;' \
"
