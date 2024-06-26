import json
import os
import io
import sys


def forward(
    input_json,
    output_conf,
):
    with io.open(
        input_json,
        'r'
    ) as f:
        forward_nginx = json.load(f)

    with io.open(
        output_conf,
        'w'
    ) as f:
        names = [o['app_name'] for o in forward_nginx]

        if not '' in names:
            forward_nginx.append(
                dict(
                    app_name='',
                    redirect_url='https://product-development-service.blogspot.com',
                )
            )

        sections = dict()

        for entry in forward_nginx:
            location_path = None

            if entry['app_name'] != '':
                location_path = '/%s/' % entry['app_name']
            else:
                location_path = '/'

            if 'server_name' in entry:
                server_name = entry['server_name']
            else:
                server_name = 'default_server'

            if not server_name in sections:
                sections[server_name] = []


            location_get = lambda location_body, location_path2, prefix=None,: (
                r'''
  location {location} {
    {location_body}
  }
                '''.replace(
                    '{location_body}', location_body,
                ).replace(
                    '{location}', '%s  %s' % (
                        (
                            '^~'
                            if prefix is None
                            else prefix
                        ),
                        location_path2
                    ),
                )
            )

            if 'target_endpoint' in entry:
                location_body_get = lambda target_endpoint: \
                    r'''
    proxy_set_header Host $http_host;
    proxy_set_header X-Forwarded-For $t1;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection $connection_upgrade;
    proxy_redirect off;
    proxy_buffering off;
    proxy_pass {target_endpoint};
                    '''.replace(
                        '{target_endpoint}', target_endpoint,
                    )

                if 'fallback_endpoint' in entry:
                    fallback_name = '@fallback-%s' % os.urandom(10).hex()

                    sections[server_name].append(
                        location_get(
                            location_body_get(entry['target_endpoint']) + r'''
    proxy_intercept_errors on;
    error_page 502 =200 {fallback_name};
                            '''.replace(
                                '{fallback_name}', fallback_name,
                            ),
                            location_path,
                        )
                    )

                    sections[server_name].append(
                        location_get(
                            location_body_get(entry['fallback_endpoint']),
                            fallback_name,
                            '',
                        )
                    )
                else:
                    sections[server_name].append(
                        location_get(
                            location_body_get(entry['target_endpoint']),
                            location_path,
                        )
                    )
            elif 'redirect_url' in entry:
                sections[server_name].append(
                    location_get(
                        r'''
      return 302 {redirect_url}$request_uri;
                        '''.replace(
                            '{redirect_url}', entry['redirect_url'],
                        ),
                        location_path,
                    )
                )
            else:
                raise NotImplementedError

        servers = []

        for server_name, current_sections in sections.items():
            servers.append(
                r'''
server {
  set $t1 $remote_addr;
  if ($http_x_forwarded_for)
  {
    set $t1 $http_x_forwarded_for;
  }

  server_name {server_name};
  listen 80 {default_server};
  client_max_body_size 50M;

  {sections_config}
}
                '''.replace(
                  '{sections_config}', '\n'.join(current_sections)
                ).replace(
                  '{server_name}',
                  (
                    '_'
                    if server_name == 'default_server'
                    else server_name
                  ),
                ).replace(
                  '{default_server}',
                  (
                    ''
                    if not server_name == 'default_server'
                    else server_name
                  )
                )
            )

        f.write(r'''
events {
  multi_accept on;
  worker_connections 64;
}

http {
  log_format main
    '[$time_local][$remote_addr:$remote_port, $http_x_forwarded_for, $t1, $http_host]'
    '[$request_length,$bytes_sent,$request_time]'
    '[$status][$request]'
    '[$http_user_agent][$http_referer]';

  access_log /dev/null combined;
  access_log /dev/stderr main;
  gzip on;
  server_tokens off;

  {servers_config}

  map $http_upgrade $connection_upgrade {
    default upgrade;
    '' close;
  }
}
        '''.replace(
          '{servers_config}', '\n'.join(servers)
        ))

def ssl(input_json, output_conf):
    with io.open(
        input_json,
        'r'
    ) as f:
        ssl_nginx = json.load(f)

    servers = []

    if 'default_server' in ssl_nginx:
        server = ssl_nginx['default_server']

        servers.append(
            r'''
server {
  set $t1 $remote_addr;
  if ($http_x_forwarded_for)
  {
    set $t1 $http_x_forwarded_for;
  }

  listen 443 ssl default_server;
  server_name _;

  client_max_body_size {client_max_body_size};

  ssl_certificate {signed_chain_cert};
  ssl_certificate_key {domain_key};

  return 444;
}
            '''.replace(
                  '{signed_chain_cert}', server['signed_chain_cert'],
              ).replace(
                  '{client_max_body_size}', server['client_max_body_size'],
              ).replace(
                  '{domain_key}', server['domain_key'],
              )
        )

    for server in ssl_nginx['servers']:
        servers.append(
            r'''


server {
  set $t1 $remote_addr;
  if ($http_x_forwarded_for)
  {
    set $t1 $http_x_forwarded_for;
  }

  listen 80;
  server_name {server_names};
  client_max_body_size {client_max_body_size};

  location ~ ^/.well-known/acme-challenge/ {
    alias /var/www/;
    try_files $uri =404;
  }

  location ~ {
    #return 444;
    return 301 https://$host$request_uri;
  }
}

server {
  set $t1 $remote_addr;
  if ($http_x_forwarded_for)
  {
    set $t1 $http_x_forwarded_for;
  }

  listen 443 ssl;
  server_name {server_names};

  client_max_body_size {client_max_body_size};

  ssl_certificate {signed_chain_cert};
  ssl_certificate_key {domain_key};

  location ^~ / {
    proxy_set_header Host $http_host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection $connection_upgrade;
    proxy_redirect off;
    proxy_buffering off;
    proxy_pass http://app:80;
  }
}
            '''.replace(
                '{server_names}', ' '.join(server['server_names'])
              ).replace(
                  '{signed_chain_cert}', server['signed_chain_cert'],
              ).replace(
                  '{client_max_body_size}', server['client_max_body_size'],
              ).replace(
                  '{domain_key}', server['domain_key'],
              )
        )

    with io.open(
        output_conf,
        'w'
    ) as f:
        f.write(
            r'''
events {
  multi_accept on;
  worker_connections 64;
}

http {
  log_format main
  '[$time_local][$remote_addr:$remote_port, $http_x_forwarded_for, $t1, $http_host]'
  '[$request_length,$bytes_sent,$request_time]'
  '[$status][$request]'
  '[$http_user_agent][$http_referer]';

  access_log /dev/null combined;
  access_log /dev/stderr main;
  gzip on;
  server_tokens off;

  {servers}


  map $http_upgrade $connection_upgrade {
    default upgrade;
    '' close;
  }
}
            '''.replace('{servers}', '\n'.join(servers))
        )


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'ssl':
        ssl(*sys.argv[2:])
    else:
        forward(*sys.argv[1:])
