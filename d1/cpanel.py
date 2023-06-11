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
                not_alive = not (
                    requests.get(v['url'], timeout=0.5).status_code
                    == 200
                )
            except:
                logging.error(json.dumps(dict(
                    error=traceback.format_exc(),
                    time_iso=datetime.datetime.now().isoformat(),
                )))
                not_alive = True

            if not_alive:
                logging.error(json.dumps(
                    dict(
                        args=o.args,
                        k=k,
                        #o=pprint.pformat(o.__dict__),
                        status='not_alive',
                        time_iso=datetime.datetime.now().isoformat(),
                    )
                ))

                #stop_task(o)
                #del t1[k]
                continue

            if not o.poll() is None:
                logging.error(json.dumps(
                    dict(
                        #o=pprint.pformat(o.__dict__),
                        args=o.args,
                        k=k,
                        return_code=o.poll(),
                        status='crashed',
                        time_iso=datetime.datetime.now().isoformat(),
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
