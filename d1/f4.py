import subprocess
import time
import os
import datetime
import traceback
import io
import pprint

def f1():
    LOG_SIZE = 10 * 1024 * 1024
    busybox_path = '%s/tmp/busybox' % os.environ['HOME']
    log_path = '%s/p1/p1/log-f4.txt' % os.environ['HOME']
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def trim_log():
        if not os.path.exists(log_path):
            return

        log_size = 0

        try:
            log_stats = os.stat(log_path)
            log_size = log_stats.st_size
        except:
            return

        if log_size > LOG_SIZE:
            try:
                log_path2 = os.path.splitext(log_path)
                os.rename(
                    log_path,
                    '%s-backup%s' % (
                        log_path2[0],
                        log_path2[1],
                    ),
                )
            except:
                return

    def log(data=None):
        with io.open(log_path, 'a') as f:
            f.write(
                '%s\n%s\n%s\n' % (
                    datetime.datetime.now().isoformat(),
                    pprint.pformat(data),
                    traceback.format_exc(),
                )
            )

        trim_log()

    while True:
        try:
            t1 = subprocess.check_output(
                [busybox_path, 'ps', '-e', 'cpu']
            ).decode('utf-8')
            t2 = t1.splitlines()
            if len(t2) > 20:
                log(
                    dict(t2=t2)
                )
                print('\n', end='')

            print('\r%s %d' % (
                datetime.datetime.now().isoformat(),
                len(t2),
            ), end='')
        except:
            log()
        time.sleep(10)

if __name__ == '__main__':
    f1()
