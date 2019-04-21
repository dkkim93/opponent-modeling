import torch
import argparse
import os
import numpy as np
from misc.utils import set_log, make_env
from tensorboardX import SummaryWriter
from trainer.train import train


def set_policy(env, tb_writer, log, args, name, i_agent):
    if name == "opponent":
        from policy.opponent import Opponent
        policy = Opponent(env=env, log=log, name=name, args=args, i_agent=i_agent)
    else:
        raise ValueError("Invalid name")

    return policy


def main(args):
    # Create dir
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Set logs
    tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))
    log = set_log(args)

    # Create env
    env = make_env(args)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize policy
    opponent = set_policy(env, tb_writer, log, args, name="opponent", i_agent=0)

    # Start train
    train(
        opponent=opponent, 
        env=env, 
        log=log,
        tb_writer=tb_writer,
        args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # TD3 Algorithm
    parser.add_argument(
        "--discount", default=0.99, type=float, 
        help="Discount factor")
    parser.add_argument(
        "--tau", default=0.01, type=float, 
        help="Target network update rate")
    parser.add_argument(
        "--start-timesteps", default=1e4, type=int, 
        help="How many time steps purely random policy is run for")
    parser.add_argument(
        "--expl-noise", default=0.1, type=float, 
        help="Std of Gaussian exploration noise")
    parser.add_argument(
        "--batch-size", default=50, type=int, 
        help="Batch size for both actor and critic")
    parser.add_argument(
        "--policy-noise", default=0.2, type=float, 
        help="Noise added to target policy during critic update")
    parser.add_argument(
        "--noise-clip", default=0.5, type=float, 
        help="Range to clip target policy noise")
    parser.add_argument(
        "--policy-freq", default=2, type=int,
        help="Frequency of delayed policy updates")
    parser.add_argument(
        "--actor-lr", default=0.0001, type=float,
        help="Learning rate for actor")
    parser.add_argument(
        "--critic-lr", default=0.001, type=float,
        help="Learning rate for critic")
    parser.add_argument(
        "--grad-clip", default=0.5, type=float,
        help="Gradient clipping to prevent explosion")

    # Opponent
    parser.add_argument(
        "--opponent-n-hidden", default=400, type=int,
        help="Number of hidden units")

    # Env
    parser.add_argument(
        "--env-name", type=str, required=True,
        help="OpenAI gym environment name")
    parser.add_argument(
        "--ep-max-timesteps", type=int, required=True,
        help="Episode is terminated when max timestep is reached.")

    # Misc
    parser.add_argument(
        "--seed", default=0, type=int, 
        help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument(
        "--prefix", default="", type=str,
        help="Prefix for tb_writer and logging")
    parser.add_argument(
        "--save-models", action="store_true", 
        help="Whether or not models are saved")

    args = parser.parse_args()

    # Set log name
    args.log_name = \
        "env::%s_seed::%s_tau::%s_start::%s_noise_std::%s_batch_size::%s_prefix::%s_log" % (
            args.env_name, str(args.seed), args.tau, args.start_timesteps, 
            args.expl_noise, args.batch_size, args.prefix)

    main(args=args)
