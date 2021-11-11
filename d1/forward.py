import subprocess
import threading
import io
import json
import sys
import os
import pprint
import traceback
import time


def forward(config_dir, config):
    server_address = config['server_address']
    username = config['username']
    target_address = config['target_address']
    blank_endpoint = config['blank_endpoint']
    target_ports = config['target_ports']
    app_name = config['app_name']
    ssh_key = os.path.join(
        config_dir,
        config['ssh_key']
    )
    ssh_command = [
        'ssh',
        '-o', 'ExitOnForwardFailure yes',
        '-o', 'StrictHostKeyChecking no',
        '-o', 'ServerAliveInterval 1',
        '-o', 'ConnectTimeout 1',
    ]

    ports = dict(
        target=[
            *ssh_command,
            '-i', ssh_key,
            '%s@%s' % (username, server_address),
            *sum([
                [
                    '-R', '0.0.0.0:%d:%s:%d' % (
                        pair[0],
                        target_address,
                        pair[1],
                    )
                ]
                for pair in target_ports
            ], []),
            '-N'
        ],
        blank=[
            *ssh_command,
            '-i', ssh_key,
            '%s@%s' % (username, server_address),
            *sum([
                [
                    '-R', '0.0.0.0:%d:%s' % (
                        pair[0],
                        blank_endpoint,
                    )
                ]
                for pair in target_ports
            ], []),
            '-N'
        ]
    )

    has_server = lambda : subprocess.call([
        'ping','-c', '1', target_address,
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE) == 0

    #notify = lambda msg : subprocess.check_output(['notify-send', '-t', '5000', app_name, msg])
    notify = lambda msg: subprocess.call(['echo', app_name, msg])

    while True:
        notify('started')
        if has_server():
            t6 = 'target'
            notify('has_server')
        else:
            t6 = 'blank'
            notify('blank_app')
        t2 = ports[t6]
        with subprocess.Popen(t2) as p:
            try:
                while True:
                    time.sleep(10)
                    t3 = has_server()
                    t4 = None
                    if t6 == 'target' and not t3:
                        t4 = 'no server'
                    elif t6 == 'blank' and t3:
                        t4 = 'server found'
                    if not t4 is None:
                        notify(t4)
                        raise RuntimeError(t4)
                    assert p.poll() is None
            except KeyboardInterrupt:
                break
            except:
                pprint.pprint(traceback.format_exc())
                continue
            finally:
                p.terminate()
                notify('stopped')

config_path = sys.argv[1]
config_dir = os.path.split(
    config_path
)[0]

with io.open(config_path, 'r') as f:
    configs = json.load(f)

t1 = []
for config in configs:
    t2 = threading.Thread(
        target=forward,
        args=[config_dir, config],
    )
    t2.start()
    t1.append(t2)

for o in t1:
    o.join()
