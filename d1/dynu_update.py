import requests
import logging
import pprint
import copy
import io
import json
import sys

logging.warning('start dynu_update')
with io.open(
    sys.argv[1],
    'r'
) as f:
    dynu_config = json.load(f)
logging.warning('loaded dynu_config')

t2 = requests.get(
    'https://api.dynu.com/v2/oauth2/token',
    auth=(
        dynu_config['oath2_client_id'],
        dynu_config['oath2_secret'],
    )
).json()
logging.warning('got access_token')

t1 = requests.get(
    'https://api.dynu.com/v2/dns',
    headers={
        'Authorization': 'Bearer %s' % t2['access_token']
    }
).json()
DYNU_DOMAIN = t1['domains'][0]
logging.warning('got dynu_domain')

t3 = requests.get(
    'https://api.dynu.com/v2/dns/%d/webredirect' % DYNU_DOMAIN['id'],
    headers={
        'Authorization': 'Bearer %s' % t2['access_token']
    }
).json()
DYNU_REDIRECT = t3['webRedirects'][0]
logging.warning('got dynu_redirect')

NGROK_DOMAIN = sys.argv[2]
t6 = requests.get('http://%s:4040/api/tunnels' % NGROK_DOMAIN).json()
TUNNEL_URL = t6['tunnels'][0]['public_url']
logging.warning('got tunnel_url')



if TUNNEL_URL != DYNU_REDIRECT['url']:
    t5 = copy.deepcopy(t3['webRedirects'][0])
    t5.update(
        dict(
            url=TUNNEL_URL,
        )
    )

    DYNU_REDIRECT = requests.post(
        'https://api.dynu.com/v2/dns/%d/webRedirect/%d' % (
            DYNU_DOMAIN['id'],
            t3['webRedirects'][0]['id']
        ),
        headers={
            'Authorization': 'Bearer %s' % t2['access_token']
        },
        json=t5
    ).json()
    logging.warning('updated dynu_redirect')
else:
    logging.warning('skip update dynu_redirect')

logging.warning(
    pprint.pformat(
        dict(
            NGROK_DOMAIN=NGROK_DOMAIN,
            TUNNEL_URL=TUNNEL_URL,
            DYNU_DOMAIN=DYNU_DOMAIN,
            DYNU_REDIRECT=DYNU_REDIRECT,
        )
    )
)

logging.warning('done dynu_update')
