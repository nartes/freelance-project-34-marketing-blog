py3 << EOF
def f1():
    t1 = vim.current.window
    t2 = t1.width
    vim.command('vnew')
    t3 = t2 // 3
    vim.command('vertical resize %d' % t3)
    vim.current.window = t1

def f2():
    context = {
        k : vim.options['splitright']
        for k in ['splitright']
    }
    try:
        current_window = vim.current.window
        vim.options['splitright'] = True
        vim.command('vnew')
        vim.command('r! tmux show-buffer')
        vim.current.window = current_window
    finally:
        for k, v in context.items():
            vim.options[k] = v

def f5_1(pattern, flags, info):
    import subprocess
    import io
    import re
    import tempfile
    import traceback
    import logging

    #print([pattern, flags, info])
    completed_process = None

    try:
        completed_process = subprocess.run([
            'git', 'grep', '-n', '-P', pattern,
        ], capture_output=True,)
        assert (
            completed_process.returncode == 0 or
            (
                completed_process.stdout == b''
                #completed_process.stdout == b'' and
                #completed_process.stderr == b''
            )
        )
        t1 = completed_process.stdout
    except:
        logging.error(''.join([
            traceback.format_exc(),
            getattr(completed_process, 'stdout', b'').decode('utf-8'),
            getattr(completed_process, 'stderr', b'').decode('utf-8'),
        ]))
        t1 = b''

    def watch(data):
        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            with io.open(f.name, 'wb') as f2:
                f2.write(data)
            vim.command('!less %s' % f.name)

    #watch(t1)

    t2 = []
    for o in t1.splitlines():
        try:
            #watch(o.encode('utf-8'))
            t3 = o.decode('utf-8')
            t4 = re.compile(r'^([^\:\=]+)[\:\=](\d+)[\:\=](.*)$').match(t3)
            if not t4 is None:
                t2.append(
                    dict(
                        name=t4[3].strip(),
                        filename=t4[1],
                        cmd=t4[2],
                    )
                )
        except:
            pass
    #print(t2)

    #return [{'name': 'blah', 'filename': 'docker-compose.yml', 'cmd': '23'}]
    return t2
EOF

function! F5(pattern, flags, info)
  let res = py3eval(
    \'f5_1(
      \vim.bindeval("a:pattern").decode("utf-8"),
      \vim.bindeval("a:flags"),
      \vim.bindeval("a:info")
    \)'
  \)
  return res
endfunc

set tagfunc=F5
