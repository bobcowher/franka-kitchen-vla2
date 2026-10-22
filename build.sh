#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate llm_course

# python -u scripts/train_spam.py
python scripts/test_smolvlm2.py
