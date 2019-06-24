#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 

# Tensorboard
pkill tensorboard
# rm -rf logs/tb*
tensorboard --logdir logs/ &

# Virtualenv
cd $DIR
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Add baseline package to path
export PYTHONPATH=$DIR/thirdparty/multiagent-particle-envs:$PYTHONPATH

# Train tf 
print_header "Training network"
cd $DIR

# # Comment for using GPU
# export CUDA_VISIBLE_DEVICES=-1

# Experiment 1
python3.6 main.py \
--env-name "Regression-v0" \
--seed 0 \
--learner-type "finetune" \
--policy-type "continuous" \
--ep-max-timesteps 500 \
--n-traj 1 \
--meta-lr 0.01 \
--meta-batch-size 20 \
--fast-batch-size 5 \
--meta-lr 0.003 \
--fast-lr 5 \
--prefix ""
