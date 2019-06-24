import torch
import argparse
import os
import numpy as np
from misc.utils import set_log, set_policy
from tensorboardX import SummaryWriter
from misc.batch_sampler import BatchSampler


def train(sampler, meta_policy, log, tb_writer, args):
    task_id = 0

    while True:
        # Sample episode from current task
        sampler.reset_task(task_id)
        episodes = sampler.sample()

        # Test adaptation performance
        meta_policy.meta_test(task_id, episodes)

        # Add to memory
        meta_policy.add_episodes(task_id, episodes)
    
        # Train meta-parameter
        meta_policy.meta_train(task_id)
    
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
    meta_policy = set_policy(sampler, log, tb_writer, args, name="meta_policy", policy_id=0)

    # Start training
    train(
        sampler=sampler, meta_policy=meta_policy, 
        log=log, tb_writer=tb_writer, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # General
    parser.add_argument(
        "--policy-type", type=str, 
        choices=["discrete", "continuous", "normal"],
        help="Policy type available only for discrete, normal, and continuous")
    parser.add_argument(
        "--n-hidden", default=32, type=int, 
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
        "env::%s_seed::%s_prefix::%s_log" % (
            args.env_name, str(args.seed), args.prefix)

    main(args=args)
