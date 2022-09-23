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
EOF

filetype plugin indent on

set number
set noswapfile

set wrap
set textwidth=100
set colorcolumn=100
colorscheme morning

syntax on
set hls

map <Leader>w <C-w>
map <Leader>r :source ~/.vimrc<CR>:echo 'reloaded'<CR>
map <Leader>m :py3 f1()<CR>
map <Leader>cq :cq<CR>
map <Leader>i1 :set sw=4 sts=4 ts=4 et ai ci<CR>:retab<CR>
map <Leader>i2 :set sw=2 sts=2 ts=2 et ai ci<CR>:retab<CR>
set foldmethod=indent
