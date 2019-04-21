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
 #rm -rf logs/tb*
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

# Comment for using GPU
export CUDA_VISIBLE_DEVICES=-1

# Experiment 1
python3.6 main.py \
--env-name "complex_push_two" \
--tau 0.01 \
--ep-max-timesteps 100 \
--start-timesteps 2000 \
--teacher-start-timesteps 320000 \
--expl-noise 0.1 \
--batch-size 50 \
--max-timesteps 500000000 \
--seed 0 \
--session 1800 \
--n-eval 10 \
--n-teacher 2 \
--student-done \
--load-student-memory \
--prefix ""
