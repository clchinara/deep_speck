#!/bin/bash

# Generate a timestamp in the format YYYYMMDDHHMM
timestamp=$(date "+%Y%m%d%H%M")

# Execute the Python script and capture its output
python "train_${1}_rounds.py" | tee "logs_inv1_train_${1}_rounds_${timestamp}.txt"
