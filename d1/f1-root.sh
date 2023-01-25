#!/bin/sh

echo user;
whoami
echo udev rules;
sudo cp dotfiles/etc/udev/rules.d/40-leds.rules /etc/udev/rules.d/40-leds.rules
