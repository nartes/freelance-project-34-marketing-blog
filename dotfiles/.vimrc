set nocompatible
filetype off

set viminfo+=/1000000,:1000000

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
map <Leader>i1 :set sw=4 sts=4 ts=4 et ai ci<CR>:retab<CR>
map <Leader>i2 :set sw=2 sts=2 ts=2 et ai ci<CR>:retab<CR>
