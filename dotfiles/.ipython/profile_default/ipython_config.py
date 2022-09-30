c.InteractiveShellApp.exec_lines = [
    '%autoreload 2',
    r'''
def ipython_update_shortcuts():
    import IPython
    import prompt_toolkit.filters
    t1 = IPython.get_ipython()
    t2 = t1.pt_app
    t3 = [o for o in t2.key_bindings.bindings if 'f2' in repr(o).lower()]
    assert len(t3) == 1
    t4 = t3[0]
    t2.key_bindings.remove(t4.handler)
    t2.key_bindings.add(
        'e', filter=~prompt_toolkit.filters.vi_insert_mode,
    )(t4.handler)
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
