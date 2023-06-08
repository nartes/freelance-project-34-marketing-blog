import subprocess
import sys
import io
import copy
import traceback
import datetime
import pprint
import logging
import json
import time

with io.open(
    'tmp/d1/cpanel.json', 'r'
) as f:
    t3 = json.load(f)

t2 = copy.deepcopy(t3)
for k in t2:
    v = t2[k]
    v['task'] = lambda : subprocess.Popen(
        v['task_cmd'],
        stdin=subprocess.DEVNULL,
    )

def stop_task(task):
    task.terminate()
    try:
        task.wait(1)
    except:
        task.kill()

t1 = dict()

shutdown = False

while True:
    try:
        for k, v in t2.items():
            if not k in t1:
                logging.info(json.dumps(dict(
                    task=k,
                    status='starting',
                )))
                t1[k] = v['task']()
                logging.info(json.dumps(dict(
                    task=k,
                    status='started',
                )))
                continue

            o = t1[k]

            not_alive = None

            try:
                url_content = []
                with subprocess.Popen(
                    [
                        'curl',
                        '-q', '--silent',
                        '-v',
                        '--max-time', '4',
                        '--max-filesize', '%d' % (4 * 1024 * 1024),
                        v['url'],
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                ) as curl:
                    def read_chunk():
                        chunk = curl.stderr.read().decode('utf-8')
                        if isinstance(chunk, str) and len(chunk) > 0:
                            url_content.append(chunk)

                        if isinstance(chunk, str) and 'status: ' in chunk:
                            stop_task(curl)

                        return chunk

                    while curl.poll() is None:
                        read_chunk()

                    while True:
                        chunk = read_chunk()

                        if chunk is None or chunk == '':
                            break

                url_content2 = ''.join(url_content)

                if not 'status: 502' in url_content2 and (
                    'status: 200' in url_content2 or
                    'status: 302' in url_content2
                ):
                    not_alive = False
                else:
                    not_alive = True
            except:
                logging.error(json.dumps(dict(
                    error=traceback.format_exc(),
                )))
                not_alive = True

            if not_alive:
                logging.error(json.dumps(
                    dict(
                        o=pprint.pformat(o.__dict__),
                        status='not_alive',
                    )
                ))

                stop_task(o)
                del t1[k]
                continue

            if not o.poll() is None:
                logging.error(json.dumps(
                    dict(
                        o=pprint.pformat(o.__dict__),
                        return_code=o.poll(),
                        status='crashed',
                    )
                ))
                del t1[k]
                continue

        if shutdown:
            break

        print('\r%s tasks %d' % (
            datetime.datetime.now().isoformat(),
            len(t1),
        ), end='')
        sys.stdout.flush()
    except KeyboardInterrupt:
        print('\nshutting down')
        break
    finally:
        time.sleep(10)

for o in t1:
    stop_task(o)
