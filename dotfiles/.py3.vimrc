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
EOF
