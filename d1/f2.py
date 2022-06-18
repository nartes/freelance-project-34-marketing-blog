import os
import tempfile
import time
import io
import traceback
import subprocess
import json
import sys
import pprint


sys.path.insert(0, os.path.dirname(__file__))


def application(environ, start_response):
    try:
        t4 = int(environ.get('CONTENT_LENGTH', '0'))
        t3 = environ['wsgi.input'].read(t4)
        def op1(rh):
            t5 = rh.split('_')
            t6 = ['%s%s' % (o[0].upper(), o[1:].lower()) for o in t5]
            return '-'.join(t6)
        t2 = {
            op1(k[5:]) : v
            for k, v in environ.items()
            if k.startswith('HTTP_')
        }
        for k, v in environ.items():
            if k in [
                'CONTENT_TYPE',
            ]:
                t2[op1(k)] = v
        t7 = dict(
            uri=environ['REQUEST_URI'],
            method=environ['REQUEST_METHOD'],
            protocol=environ['SERVER_PROTOCOL'],
        )
        output_dat = None
        input_dat = None
        def op1():
            for o in [input_dat, output_dat]:
                if not o is None and os.path.exists(o):
                    os.unlink(o)
        try:
            output_dat = tempfile.mktemp(suffix='.dat')
            input_dat = tempfile.mktemp(suffix='.dat')
            op1()

            with io.open(input_dat, 'wb') as f:
                f.write(t3)

            t17 = [
                'curl',
                'http://127.0.0.1:9050%s' % t7['uri'],
                *sum([
                    ['--header', '%s: %s' % (k, v)]
                    for k, v in t2.items()
                ], []),
                '-X', t7['method'],
                '--data-binary', '@%s' % input_dat,
                '--max-filesize', '%d' % (60 * 1024 * 1024),
                '-o', output_dat,
                '-v',
                '-q',
            ]
            with io.open('1.json', 'w') as f:
                f.write(json.dumps(dict(t17=t17, t3=t3.decode('utf-8'))))
            with subprocess.Popen(
                t17, stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE
            ) as p:
                try:
                    p.wait(20)
                    response_headers = [
                        o[2:]
                        for o in p.stderr.read().decode('utf-8').splitlines()
                        if o.startswith('< ')
                    ]
                    t9 = '\r\n'.join(response_headers)
                    if not 'Content-Length: 0' in t9:
                        for k in range(3):
                            try:
                                with io.open(output_dat, 'rb') as f:
                                    t10 = f.read()
                                break
                            except:
                                time.sleep(0.05)
                    else:
                        t10 = b''
                finally:
                    p.terminate()
        except:
            t9 = 'FUCK SHIT\r\n'
            t10 = traceback.format_exc().encode('utf-8')
        finally:
            op1()

        if not any([o.startswith('Content-Length') for o in t9]):
            t9 = t9.replace('Transfer-Encoding: chunked\r\n', '')
            t9 += 'Content-Length: %d\r\n' % len(t10)
        t11 = t9.encode('utf-8') + t10
        t13 = t9.splitlines()[0]
        t14 = t13.find(' ')
        t15 = t13[t14 + 1:]
        t16 = [(o[:o.find(':')], o[o.find(':') + 1:]) for o in t9.splitlines()[1:]]
        if False:
            t1 = start_response('200 OK', [('Content-Type', 'text/plain')])
            t1(t15)
            t1(json.dumps(t16))
            t1(json.dumps(t17))
            t1(t10)
        else:
            t1 = start_response(t15, t16)
            t1(t10)
    except:
        with io.open('log.txt', 'a') as f:
            f.write(traceback.format_exc())

    return []
