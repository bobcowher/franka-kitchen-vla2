#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate franka-kitchen-vla2 

# python -u scripts/train_spam.py
python scripts/train.py
