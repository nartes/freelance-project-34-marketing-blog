FROM python:3
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y iputils-ping
ENTRYPOINT sh -c ' \
  python d1/forward.py tmp/cache/forward_data/forward.config.json \
'
