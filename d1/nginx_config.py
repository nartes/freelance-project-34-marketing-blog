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
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
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
  server {
    listen 80;
    client_max_body_size 50M;

    {sections_config}

    location / {
      return 302 https://product-development-service.blogspot.com;
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
