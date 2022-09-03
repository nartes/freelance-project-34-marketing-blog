import multiprocessing
import time
import traceback
import datetime
import requests
import logging
import pprint
import copy
import io
import json
import sys


def update(
    dynu_config,
):
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
    TUNNEL_URL = t6['tunnels'][0]['public_url'].replace('http://', 'https://')
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

def service():
    logging.warning('start dynu_update')

    need_wait = False
    last_disabled = False

    while True:
        try:
            if need_wait:
                need_wait = False
                time.sleep(900)

            with io.open(
                sys.argv[1],
                'r'
            ) as f:
                dynu_config = json.load(f)

            if dynu_config.get('enabled') != True:
                if not last_disabled:
                    last_disabled = True
                    logging.warning('disabled')
                need_wait = True
                continue
            else:
                last_disabled = False
                logging.warning('loaded dynu_config')

            with multiprocessing.Pool(processes=1) as pool:
                while True:
                    pool.apply(
                        update,
                        args=(
                            dynu_config,
                        )
                    )

                    time.sleep(900)
        except KeyboardInterrupt:
            break
        except:
            logging.error('%s\n%s' % (
                datetime.datetime.now(tz=datetime.timezone.utc),
                traceback.format_exc().strip()
            ))
            need_wait = True


if __name__ == '__main__':
    service()
