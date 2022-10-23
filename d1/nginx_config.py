import json
import io
import sys


with io.open(
    sys.argv[1],
    'r'
) as f:
    forward_nginx = json.load(f)

with io.open(
    sys.argv[2],
    'w'
) as f:
    sections = []
    for entry in forward_nginx:
        sections.append(r'''
    location ^~ /{app_name}/ {
      proxy_set_header Host $http_host;
      proxy_set_header X-Forwarded-For $t1;
      proxy_set_header X-Forwarded-Proto $scheme;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection $connection_upgrade;
      proxy_redirect off;
      proxy_buffering off;
      proxy_pass {target_endpoint};
    }
        '''.replace(
            '{app_name}', entry['app_name'],
        ).replace(
            '{target_endpoint}', entry['target_endpoint'],
        ))
    f.write(r'''
events {
  multi_accept on;
  worker_connections 64;
}

http {
  log_format main
    '[$time_local][$remote_addr, $http_x_forwarded_for, $t1]'
    '[$request_length,$bytes_sent,$request_time]'
    '[$status][$request]'
    '[$http_user_agent][$http_referer]';

  access_log /dev/null combined;
  access_log /dev/stderr main;

  server {
    set $t1 $remote_addr;
    if ($http_x_forwarded_for)
    {
      set $t1 $http_x_forwarded_for;
    }


    listen 80;
    client_max_body_size 50M;

    {sections_config}

    location / {
      return 302 https://product-development-service.blogspot.com$request_uri;
    }
  }

  map $http_upgrade $connection_upgrade {
    default upgrade;
    '' close;
  }
}
    '''.replace(
      '{sections_config}', '\n'.join(sections)
    ))
