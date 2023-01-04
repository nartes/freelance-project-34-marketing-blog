# The following lines were added by compinstall

zstyle ':completion:*' completer _expand _complete _ignored _correct _approximate
zstyle :compinstall filename '~/.zshrc'

setopt INC_APPEND_HISTORY SHARE_HISTORY AUTO_PUSHD PUSHD_IGNORE_DUPS

autoload -Uz compinit
compinit
# End of lines added by compinstall
# Lines configured by zsh-newuser-install
HISTFILE=~/.histfile
HISTSIZE=1000000
SAVEHIST=1000000
# End of lines configured by zsh-newuser-install

bindkey -d
bindkey -v


eval `keychain --eval --quiet`

if [[ $TTY == "/dev/tty1" ]] {
  #exec startx
  exec sh -c $'sway 2>&1 | logger -d --tag sway --id=$(pgrep  -i \'sway$\');'
}
