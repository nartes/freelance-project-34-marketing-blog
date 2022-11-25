import os
import select
import socket
import copy
import re
import shutil
import datetime
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
    MAX_FILE_SIZE = 512 * 1024 * 1024
    LOG_SIZE = 10 * 1024 * 1024
    MAX_TIME = 16
    EASTER_TOKEN = 'FUCK'

    def __init__(self, environ, start_response):
        self.environ = environ

        try:
            with io.open(
                'wsgi_config.json',
                'r'
            ) as f:
                config = json.load(f)

            assert isinstance(config, dict)
        except:
            config = dict()

        self.config = config
        self.start_response = start_response
        self.log_path = 'log.txt'

    def debug(self):
        debug_token = self.config.get('debug_token')
        if not (
            not debug_token is None and \
            isinstance(debug_token, str)
        ):
            return False

        return self.environ.get('HTTP_%s' % Application.EASTER_TOKEN) == \
            self.config.get('debug_token') or \
            ('%s=%s' % (Application.EASTER_TOKEN, debug_token)) in \
            self.environ['QUERY_STRING']

    def trim_log(self):
        if not os.path.exists(self.log_path):
            return

        log_size = 0

        try:
            log_stats = os.stat(self.log_path)
            log_size = log_stats.st_size
        except:
            return

        if log_size > Application.LOG_SIZE:
            try:
                log_path2 = os.path.splitext(self.log_path)
                os.rename(
                    self.log_path,
                    '%s-backup%s' % (
                        log_path2[0],
                        log_path2[1],
                    ),
                )
            except:
                return

    def op1(self, data=None, json_data=None,):
        if not self.debug():
            return

        if data is None:
            if not json_data is None:
                data = json.dumps(json_data)
            else:
                data = traceback.format_exc()

        self.trim_log()

        with io.open(
            self.log_path,
            'a'
        ) as f:
            f.write(
                '[%d] %s\n%s\n' % (
                    os.getpid(),
                    datetime.datetime.now().isoformat(),
                    data,
                )
            )

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

    def op10(
        self,
        headers_text=None,
        headers_lines=None,
        as_dict=None,
        detailed=None,
    ):
        if detailed is None:
            detailed = True

        if headers_text is None:
            headers_text = ''.join(
                [
                    '%s\r\n' % o
                    for o in headers_lines
                ]
            )

        if as_dict is None:
            as_dict = False

        headers_list = [
            (o[:o.find(':')], o[o.find(':') + 2:])
            for o in headers_text.splitlines()[1:]
            if len(o) > 0
        ]

        if as_dict:
            headers = dict(headers_list)
        else:
            headers = headers_list

        if detailed:
            self.op1(dict(headers_text=headers_text))
            return dict(
                first_line='%s\r\n' % headers_text.splitlines()[0],
                headers=headers,
                last_line='%s\r\n' % headers_text.splitlines()[-1]
            )
        else:
            return headers

    def op11(
        self,
        headers_dict,
        old_headers_text,
        protocol=None,
        status_suffix=None,
    ):
        first_line_split = lambda : \
            old_headers_text.splitlines()[0].split(' ', maxsplit=1)
        if protocol is None:
            protocol = first_line_split()[0]

        if status_suffix is None:
            status_suffix = first_line_split()[1]

        return ''.join(sum([
            [
                '%s %s\r\n' % (protocol, status_suffix),
            ],
            [
                '%s: %s\r\n' % (k, v)
                for k, v in headers_dict.items()
            ]
        ], []))

    def op5(
        self,
        status_code=None,
        status_text=None,
        content_type=None,
        content=None,
        headers_text=None,
    ):
        content_length = None
        if content is None:
            content = b''

        if isinstance(content, bytes):
            content_length = len(content)

        if not headers_text is None:
            if False:
                if not any([o.startswith('Content-Length') for o in headers_text]):
                    headers_text = \
                        headers_text.replace(
                            'Transfer-Encoding: chunked\r\n',
                            ''
                        )
                    headers_text += 'Content-Length: %d\r\n' % len(content)

            t13 = headers_text.splitlines()[0]
            t14 = t13.find(' ')
            t15 = t13[t14 + 1:]
            status_suffix = t15
            headers = self.op10(headers_text)
        else:
            if status_code is None:
                status_code = 200
            if status_text is None:
                status_text = 'OK'

            if content_type is None:
                content_type = 'text/plain'

            status_suffix = '%d %s' % (status_code, status_text)
            headers = [('Content-Type', content_type)]

        self.op1(json_data=dict(
            headers_text=headers_text,
            content_length=content_length,
            content_type=content_type,
            status_code=status_code,
            status_text=status_text,
            status_suffix=status_suffix,
        ))

        if isinstance(content, bytes):
            t1 = self.start_response(
                status_suffix,
                headers,
            )
            t1(content)

            return []
        else:
            headers_lines = []
            content_chunks = []
            returncode = None
            for chunk in content:
                current_headers = chunk[0]
                headers_lines.extend(current_headers)
                if len(chunk[1]) > 0:
                    content_chunks.append(chunk[1])
                returncode = chunk[2]

                self.op1(
                    json_data=[
                        headers_lines,
                        returncode,
                        len(chunk[1]),
                        len(content_chunks),
                    ]
                )
                if len(headers_lines) > 0 and headers_lines[-1] == '':
                    break

            if not returncode is None and returncode != 0:
                self.op1(
                    json_data=dict(
                        headers_lines=headers_lines,
                        returncode=returncode,
                    )
                )

            output_stream = self.environ['passenger.hijack'](True)

            headers_detailed = self.op10(
                headers_lines=headers_lines,
                detailed=True,
                as_dict=True,
            )

            self.op1(
                json_data=dict(
                    headers_detailed=headers_detailed,
                )
            )

            get_output_length = lambda : sum([len(o) for o in content_chunks])
            finished_output = False
            first_output = False
            sent_bytes = 0

            content_length = None
            content_length2 = None

            def dump_headers():
                self.op1(
                    json_data=dict(
                        aciton='dump_headers',
                        sent_byte=sent_bytes,
                        output_length=get_output_length()
                    )
                )

                nonlocal content_length
                nonlocal content_length2

                if 'Content-Length' in headers_detailed['headers']:
                    content_length2 = int(headers_detailed['headers']['Content-Length'])
                else:
                    content_length2 = 0

                if headers_detailed['headers'].get('Transfer-Encoding') == 'chunked':
                    del headers_detailed['headers']['Transfer-Encoding']
                    assert sent_bytes == 0

                    if finished_output:
                        content_length2 = get_output_length()
                    else:
                        content_length2 = Application.MAX_FILE_SIZE

                    headers_detailed['headers']['Content-Length'] = \
                        '%d' % content_length2
                    self.op1(
                        json_data=dict(
                            headers_detailed=headers_detailed,
                        )
                    )
                elif content_length2 > 512 * 1024:
                    content_length = min(2 * 1024 * 1024, content_length2)

                    #headers_detailed['headers']['Content-Length'] = '%d' % content_length
                    content_range_header = headers_detailed['headers'].get('Content-Range')
                    if not content_range_header is None and False:
                        parsed_range = re.compile(r'(\w+) (\d+)-(\d+)\/(\d+)').match(
                            content_range_header
                        )
                        assert parsed_range[1] == 'bytes'
                        start = int(parsed_range[2])
                        total = int(parsed_range[3])
                        end = start + content_length

                        headers_detailed['headers']['Content-Length'] = \
                            '%d' % content_length
                        headers_detailed['first_line'] = 'HTTP/1.1 206 Partial Content'
                        headers_detailed['headers']['Status'] = '206 Partial Content'
                        headers_detailed['headers']['Content-Range'] = \
                            'bytes %d-%d/%d' % (
                                start, end - 1, total,
                            )

                if content_length is None:
                    content_length = content_length2

                headers_detailed['headers']['Connection'] = 'close'

                output_stream.sendall(
                    headers_detailed['first_line'].encode('latin-1')
                )

                for k, v in headers_detailed['headers'].items():
                    output_stream.sendall(
                        (
                            '%s: %s\r\n' % (k, v)
                        ).encode('latin-1')
                    )
                output_stream.sendall(
                    headers_detailed['last_line'].encode('latin-1')
                )

            while True:
                if not finished_output:
                    try:
                        #self.op1(json_data=dict(action='fetch-chunk-started'))
                        chunk_detailed = next(content)
                        #self.op1(
                        #    json_data=dict(
                        #        action='fetch-chunk-done',
                        #        returncode=chunk_detailed[2],
                        #        stderr=chunk_detailed[0],
                        #        chunk_length=len(chunk_detailed[1]),
                        #    )
                        #)
                        if (len(chunk_detailed[1]) > 0):
                            content_chunks.append(chunk_detailed[1])
                    except StopIteration:
                        finished_output = True

                if (
                    not finished_output and get_output_length() > 2048 or \
                    finished_output
                ) and not first_output:
                    dump_headers()
                    first_output = True

                if first_output and len(content_chunks) > 0:
                    chunk = content_chunks[0]
                    del content_chunks[0]
                    if sent_bytes + len(chunk) > content_length:
                        chunk2 = chunk[:content_length - sent_bytes]
                        finished_output = True
                    else:
                        chunk2 = chunk

                    if len(chunk2) > 0:
                        output_stream.sendall(chunk2)
                        sent_bytes += len(chunk2)
                        #self.op1(
                        #    json_data=dict(sent_bytes=sent_bytes)
                        #)

                if finished_output and first_output and len(content_chunks) == 0:
                    break

            try:
                output_stream.shutdown(socket.SHUT_WR)
            except:
                output_stream.close()

            self.op1(json_data=dict(
                action='close_connection',
                extra=pprint.pformat(dict(
                    content_length2=content_length2,
                    sent_bytes=sent_bytes,
                )),
            ))
            return

    def op9(
        self,
        cmd_args,
        headers=None,
        input_content=None,
        input_content_length=None,
    ):
        assert isinstance(input_content_length, int) and input_content_length >= 0

        if not headers is None:
            extra_cmd_args = sum([
                ['--header', '%s: %s' % (k, v)]
                for k, v in headers.items()
            ], [])
        else:
            extra_cmd_args = []

        cmd_args2 = cmd_args + extra_cmd_args

        self.op1(json.dumps(cmd_args2))

        with subprocess.Popen(
            cmd_args2,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE
        ) as p:
            response_headers = []
            returncode = None

            sent_bytes = 0
            while sent_bytes < input_content_length and p.poll() is None:
                input_chunk = input_content.read(
                    min(1024, input_content_length - sent_bytes)
                )
                p.stdin.write(
                    input_chunk,
                )
                p.stdin.flush()

                sent_bytes += len(input_chunk)
            p.stdin.close()

            try:
                stdout_length = 0
                last_header = None

                while True:
                    if last_header == '\r\n' and len(response_headers) > 0 and \
                        'HTTP/1.1 100 Continue' in response_headers[0]:
                        self.op1(
                            json_data=dict(
                                last_header=last_header,
                                response_headers=response_headers,
                                action='restart_reading',
                            )
                        )
                        last_header = None
                        del response_headers[:]

                    if last_header == '\r\n' or not p.poll() is None:
                        break

                    last_header = p.stdout.readline().decode('utf-8')
                    response_headers.append(
                        last_header.rstrip('\r\n')
                    )

                yield (response_headers, b'', None)

                while True:
                    returncode = p.poll()

                    stdout = p.stdout.read(1024)
                    stdout_length += len(stdout)

                    yield ([], stdout, returncode)

                    if len(stdout) == 0 and not returncode is None:
                        break

                assert returncode == 0
            except Exception as exception:
                self.op1(json_data=dict(
                    stdout_length=stdout_length,
                    response_headers=response_headers,
                    returncode=returncode,
                ))
                raise exception
            finally:
                p.terminate()

    def op6(
        self,
        cmd_args,
        headers=None,
        input_content=None,
        input_content_length=None,
    ):
        return self.op9(
            cmd_args,
            headers,
            input_content=input_content,
            input_content_length=input_content_length,
        )

    def op7(self):
        try:
            with io.open(
                os.path.join(
                    os.environ['HOME'],
                    'proxy.json'
                ),
                'r'
            ) as f:
                return numpy.random.choice(json.load(f))
        except:
            with io.open('log.txt', 'a') as f:
                f.write(traceback.format_exc())

            return '127.0.0.1:9050'

    def op8(
        self,
        input_content,
        input_content_length,
        headers,
        uri,
        method
    ):
        try:
            self.op4()

            proxy_url = self.op7()

            self.op1(json_data=headers)

            t17 = [
                'curl',
                'http://%s%s' % (proxy_url, uri),
                '-g',
                '-X', method,
                '--no-buffer',
                '-i',
                '--silent',
                '--data-binary', '@-',
                '--max-filesize', '%d' % Application.MAX_FILE_SIZE,
                '--max-time', '%d' % Application.MAX_TIME,
                #'-o', self.output_dat,
                '-q',
            ]

            return self.op6(
                t17,
                headers=headers,
                input_content=input_content,
                input_content_length=input_content_length,
            )
        finally:
            self.op3()

    def run(self):
        self.op1(
            data=pprint.pformat(
                dict(
                    environ=self.environ
                )
            )
        )

        if self.environ.get('HTTP_%s' % Application.EASTER_TOKEN) == 'fuck':
            output_stream = self.environ['passenger.hijack'](True)

            content = 'fuck off'
            output_stream.sendall(
                ''.join([
                    'HTTP/1.1 200\r\n',
                    'Status: 200\r\n',
                    'Connection-Length: %d\r\n' % (len(content) + 10),
                    'Connection: close\r\n',
                    '\r\n',
                ]).encode('latin-1')
            )
            output_stream.sendall(content.encode('utf-8'))
            output_stream.shutdown(socket.SHUT_WR)
            #self.start_response(200, [])
            #return []
            return None

        try:
            t4 = int(self.environ.get('CONTENT_LENGTH', '0'))
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
                input_content=self.environ['wsgi.input'],
                input_content_length=t4,
                headers=t2,
                uri=t7['uri'],
                method=t7['method'],
            )

            return self.op5(
                content=o_8,
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
