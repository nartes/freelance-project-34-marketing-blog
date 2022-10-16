import os
import tempfile
import time
import numpy
import io
import traceback
import subprocess
import json
import sys
import pprint


sys.path.insert(0, os.path.dirname(__file__))


class Application:
    def __init__(self, environ, start_response):
        self.environ = environ
        self.start_response = start_response

    def op1(self, data=None):
        if data is None:
            data = traceback.format_exc()
        with io.open('log.txt', 'a') as f:
            f.write(data)

    def op2(self, rh):
        t5 = rh.split('_')
        t6 = ['%s%s' % (o[0].upper(), o[1:].lower()) for o in t5]
        return '-'.join(t6)

    def op3(self,):
        for o in [self.input_dat, self.output_dat]:
            if not o is None and os.path.exists(o):
                os.unlink(o)

    def op4(self,):
        self.output_dat = tempfile.mktemp(suffix='.dat')
        self.input_dat = tempfile.mktemp(suffix='.dat')

    def op5(
        status_code=None,
        status_text=None,
        content_type=None,
        content=None,
        headers_text=None,
    ):
        if not headers_text is None:
            if not any([o.startswith('Content-Length') for o in headers_text]):
                headers_text = \
                    headers_text.replace(
                        'Transfer-Encoding: chunked\r\n',
                        ''
                    )
                headers_text += 'Content-Length: %d\r\n' % len(t10)

            t13 = headers_text.splitlines()[0]
            t14 = t13.find(' ')
            t15 = t13[t14 + 1:]
            status_suffix = t15
            headers = [
                (o[:o.find(':')], o[o.find(':') + 1:])
                for o in headers_text.splitlines()[1:]
            ]
        else:
            if status_code is None:
                status_code = 200
            if status_text is None:
                status_text = 'OK'

            if content_type is None:
                content_type = 'text/plain'

            status_suffix = '%d %s' % (status_code, status_text)
            headers = [('Content-Type', content_type)]

        t1 = self.start_response(
            status_suffix,
            headers,
        )
        assert isinstance(content, bytes)

        t1(content)

        return []

    def op6(self, cmd_args,):
        self.op1(json.dumps(cmd_args))

        with subprocess.Popen(
            cmd_args,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE
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
                    for k in range(100):
                        try:
                            with io.open(self.output_dat, 'rb') as f:
                                t10 = f.read()
                            break
                        except:
                            time.sleep(0.05)
                else:
                    t10 = b''
            finally:
                p.terminate()

        return dict(
            headers_text=t9,
            content=t10
        )

    def op7(self):
        try:
            with io.open(
                os.path.join(
                    os.self.environ['HOME'],
                    'proxy.json'
                ),
                'r'
            ) as f:
                return numpy.random.choice(json.load(f))
        except:
            with io.open('log.txt', 'a') as f:
                f.write(traceback.format_exc())

            return '127.0.0.1:9050'

    def op8(self, input_content, headers, uri, method):
        try:
            self.op4()

            with io.open(
                self.input_dat,
                'wb'
            ) as f:
                f.write(input_content)


            proxy_url = self.op7()

            t17 = [
                'curl',
                'http://%s%s' % (proxy_url, uri),
                *sum([
                    ['--header', '%s: %s' % (k, v)]
                    for k, v in t2.items()
                ], []),
                '-X', method,
                '--data-binary', '@%s' % self.input_dat,
                '--max-filesize', '%d' % (60 * 1024 * 1024),
                '-o', self.output_dat,
                '-v',
                '-q',
            ]

            return self.op6(t17)
        finally:
            self.op3()

    def run(self):
        try:
            t4 = int(self.environ.get('CONTENT_LENGTH', '0'))
            t3 = self.environ['wsgi.input'].read(t4)
            t2 = {
                self.op2(k[5:]) : v
                for k, v in self.environ.items()
                if k.startswith('HTTP_')
            }
            for k, v in self.environ.items():
                if k in [
                    'CONTENT_TYPE',
                ]:
                    t2[self.op2(k)] = v

            t7 = dict(
                uri=self.environ['REQUEST_URI'],
                method=self.environ['REQUEST_METHOD'],
                protocol=self.environ['SERVER_PROTOCOL'],
            )

            o_8 = self.op8(
                t3,
                headers=t2,
                uri=t7['uri'],
                method=t7['method'],
            )

            return self.op5(
                **o_8,
            )
        except:
            self.op1()
            return self.op5(
                content='internal server error',
            )


def application(environ, start_response):
    return Application(
        environ=environ,
        start_response=start_response,
    ).run()
