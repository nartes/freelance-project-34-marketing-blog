c.InteractiveShellApp.exec_lines = [
    '%autoreload 2',
    r'''
def ipython_update_shortcuts():
    import IPython
    import prompt_toolkit.filters
    import prompt_toolkit.document
    import functools
    import tempfile
    import io
    import subprocess

    def ipython_edit_in_vim(*args, pt_app):
        content = pt_app.app.current_buffer.document.text
        lines_count = lambda text: len(text.splitlines())

        with tempfile.NamedTemporaryFile(
            suffix='.py',
            mode='w',
        ) as f:
            with io.open(f.name, 'w') as f2:
                f2.write(content)
                f2.flush()

            result = subprocess.call([
                'vim',
                '+%d' % lines_count(content),
                f.name,
            ])

            if result != 0:
                return

            f.seek(0, io.SEEK_SET)

            with io.open(f.name, 'r') as f2:
                new_content = f2.read()

            pt_app.app.current_buffer.document = \
                prompt_toolkit.document.Document(
                    new_content,
                    cursor_position=len(new_content.rstrip()),
                )

    t1 = IPython.get_ipython()
    t2 = t1.pt_app
    t3 = [o for o in t2.key_bindings.bindings if 'f2' in repr(o.keys).lower()]
    assert len(t3) == 1
    t4 = t3[0]
    t2.key_bindings.remove(t4.handler)
    t2.key_bindings.add(
        'e', filter=~prompt_toolkit.filters.vi_insert_mode,
    )(
        functools.partial(
            ipython_edit_in_vim,
            pt_app=t2,
        )
        #t4.handler
    )
    ''',
    'ipython_update_shortcuts()',
]
c.InteractiveShellApp.extensions = ['autoreload']
c.InteractiveShell.history_length = 100 * 1000 * 1000
c.InteractiveShell.history_load_length = 100 * 1000 * 1000
c.InteractiveShell.pdb = True
c.TerminalInteractiveShell.editing_mode = 'vi'
c.TerminalInteractiveShell.modal_cursor = False
c.TerminalInteractiveShell.emacs_bindings_in_vi_insert_mode = False
