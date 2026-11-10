#!/bin/bash
# Run the full pipeline: split check, training, evaluation.
set -e
python3 prepare.py
python3 train.py
python3 evaluate.py
