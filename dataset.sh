#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/gym_collision_avoidance/experiments/utils.sh

# Train tf 
print_header "Running dataset generation python script"

# # Comment for using GPU
# export CUDA_VISIBLE_DEVICES=-1

# Experiment
#cd $DIR
python create_dataset.py