#!/bin/bash
export PATH="$HOME/.pyenv/bin:$PATH"
echo "$1"
#cd ~/Documents/Projects/assistive-gym
eval "$(pyenv init -)"
python3 realtime_train.py --person-id "$1" --gender "$2" --smpl-file  "$3"