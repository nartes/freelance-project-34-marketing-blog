set nocompatible
filetype off

set viminfo+=/1000000,:1000000


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

filetype plugin indent on

set number
set noswapfile

set wrap
set textwidth=100
set colorcolumn=100
set backspace=indent,eol,start
colorscheme morning

syntax on
set hls
set term=xterm-256color

map <Leader>w <C-w>
map <Leader>wo :py3 print('fuck')<CR>
map <Leader>z :wqa<CR>
map <Leader>m :py3 f1()<CR>
map <Leader>r :redraw!<CR>
map <Leader>cq :cq<CR>
map <Leader>f2 :py3 f2()<CR>
map <Leader>f3 :source ~/.vimrc<CR>:echo 'reloaded'<CR>
map <Leader>f4 :set termguicolors<CR>
map <Leader>qy :q!<CR>
map <Leader>cq :cq1<CR>
map <Leader>dq :cq2<CR>
map <Leader>i1 :set sw=4 sts=4 ts=4 et ai ci<CR>:retab<CR>
map <Leader>i2 :set sw=2 sts=2 ts=2 et ai ci<CR>:retab<CR>
map <Leader>i3 :set t_Co=0 so=999<CR>
map <Leader>i4 :set t_Co=256 so=0<CR>
set foldmethod=indent
set nofoldenable
map <Leader>e :e #<cR>
set mouse=a
au FileType netrw nmap <buffer> <LeftMouse> <LeftMouse> <CR>
