import torch
import argparse
import os
import numpy as np
from misc.utils import set_log, set_policy, set_learner
from tensorboardX import SummaryWriter
from misc.batch_sampler import BatchSampler


def train(sampler, learner, log, tb_writer, args):
    task_id = 0

    while True:
        # Sample episode from current task
        sampler.reset_task(task_id)
        episodes = sampler.sample()

        # Test adaptation performance
        learner.test(task_id, episodes)

        # Add to memory
        learner.add_episodes(task_id, episodes)
    
        # Train meta-parameter
        learner.train(task_id)
    
        # Move to next task
        task_id += 1


def main(args):
    # Create dir
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Set logs
    tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))
    log = set_log(args)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Prepare for training
    sampler = BatchSampler(args)
    policy = set_policy(sampler, log, tb_writer, args, policy_id=0)
    learner = set_learner(policy, sampler, log, tb_writer, args, learner_id=0)

    # Start training
    train(
        sampler=sampler, learner=learner, 
        log=log, tb_writer=tb_writer, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # General
    parser.add_argument(
        "--policy-type", type=str, 
        choices=["discrete", "continuous", "normal"],
        help="Policy type available only for discrete, normal, and continuous")
    parser.add_argument(
        "--learner-type", type=str, 
        choices=["meta", "finetune"],
        help="Learner type available only for meta, finetune")
    parser.add_argument(
        "--n-hidden", default=64, type=int, 
        help="Number of hidden units")
    parser.add_argument(
        "--n-traj", default=1, type=int, 
        help="Number of trajectory to collect from each task")

    # Meta-learning
    parser.add_argument(
        "--meta-batch-size", default=25, type=int, 
        help="Number of tasks to sample for meta parameter update")
    parser.add_argument(
        "--fast-batch-size", default=5, type=int, 
        help="Number of updates for adaptation")
    parser.add_argument(
        "--meta-lr", default=0.03, type=float, 
        help="Meta learning rate")
    parser.add_argument(
        "--fast-lr", default=10.0, type=float, 
        help="Adaptation learning rate")
    parser.add_argument(
        "--first-order", action="store_true", 
        help="Adaptation learning rate")

    # Env
    parser.add_argument(
        "--env-name", default="", type=str,
        help="OpenAI gym environment name")
    parser.add_argument(
        "--ep-max-timesteps", default=10, type=int,
        help="Episode is terminated when max timestep is reached.")

    # Misc
    parser.add_argument(
        "--seed", default=0, type=int, 
        help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument(
        "--prefix", default="", type=str,
        help="Prefix for tb_writer and logging")

    args = parser.parse_args()

    # Set log name
    args.log_name = \
        "env::%s_seed::%s_learner_type::%s_meta_batch_size::%s_meta_lr::%s_fast_batch_size::%s_" \
        "fast_lr::%s_prefix::%s_log" % (
            args.env_name, str(args.seed), args.learner_type, args.meta_batch_size, args.meta_lr, 
            args.fast_batch_size, args.fast_lr, args.prefix)

    main(args=args)
