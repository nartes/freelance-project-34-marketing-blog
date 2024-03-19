#!/bin/sh

mkdir -p ~/.local/bin
cp dotfiles/.local/bin/commands ~/.local/bin/commands
mkdir -p ~/.sway
cp dotfiles/.sway/config ~/.sway/config
cp dotfiles/.zshenv ~/.zshenv
cp dotfiles/.zshrc ~/.zshrc
cp dotfiles/.vimrc ~/.vimrc
cp dotfiles/.py3.vimrc ~/.py3.vimrc
cp dotfiles/.tmux.conf ~/.tmux.conf
cp -rp \
    dotfiles/.ipython/profile_default/ipython_config.py \
    ~/.ipython/profile_default/ipython_config.py
