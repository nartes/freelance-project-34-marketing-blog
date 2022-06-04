#!/bin/sh

mkdir -p ~/.local/bin
cp dotfiles/.local/bin/commands ~/.local/bin/commands
mkdir -p ~/.sway
cp dotfiles/.sway/config ~/.sway/config
cp dotfiles/.zshenv ~/.zshenv
cp dotfiles/.tmux.conf ~/.tmux.conf
