import torch
import argparse
import os
import numpy as np
from misc.utils import set_log, make_env
from tensorboardX import SummaryWriter
from trainer.train import train


def set_policy(env, tb_writer, log, args, name, i_agent):
    if name == "student":
        from policy.student import Student
        policy = Student(env=env, log=log, name=name, args=args, i_agent=i_agent)
    elif name == "teacher":
        from policy.teacher import Teacher
        policy = Teacher(env=env, tb_writer=tb_writer, log=log, name=name, args=args, i_agent=i_agent)
    elif name == "asker":
        from policy.asker import Asker
        policy = Asker(env=env, tb_writer=tb_writer, log=log, name=name, args=args, i_agent=i_agent)
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
    student_n = [
        set_policy(env, tb_writer, log, args, name="student", i_agent=i_agent)
        for i_agent in range(args.n_student)]
    teacher_n = [
        set_policy(env, tb_writer, log, args, name="teacher", i_agent=i_agent)
        for i_agent in range(args.n_teacher)]
    asker_n = [
        set_policy(env, tb_writer, log, args, name="asker", i_agent=i_agent)
        for i_agent in range(args.n_teacher)]

    # Start train
    train(
        student_n=student_n, 
        teacher_n=teacher_n,
        asker_n=asker_n,
        env=env, 
        log=log,
        tb_writer=tb_writer,
        args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Algorithm
    parser.add_argument(
        "--discount", default=0.99, type=float, 
        help="Discount factor")
    parser.add_argument(
        "--tau", default=0.1, type=float, 
        help="Target network update rate")
    parser.add_argument(
        "--start-timesteps", default=1e4, type=int, 
        help="How many time steps purely random policy is run for")
    parser.add_argument(
        "--max-timesteps", default=1e6, type=float, 
        help="Max time steps to run environment for")
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
    parser.add_argument(
        "--n-eval", default=10, type=int,
        help="# of evaluation")

    # Student
    parser.add_argument(
        "--n-student", default=2, type=int,
        help="Number of students")
    parser.add_argument(
        "--student-n-hidden", default=400, type=int,
        help="Number of hidden units")
    parser.add_argument(
        "--student-done", action="store_true",
        help="Set student done or not")
    parser.add_argument(
        "--meta-done", action="store_true",
        help="Set student done or not")
    parser.add_argument(
        "--load-student-memory", action="store_true",
        help="Load student memory or not")

    # Teacher
    parser.add_argument(
        "--n-teacher", default=2, type=int,
        help="Number of teachers")
    parser.add_argument(
        "--session", type=int, required=True,
        help="Student reset every session")
    parser.add_argument(
        "--teacher-start-timesteps", required=True, type=int, 
        help="How many time steps purely random policy is run for")
    parser.add_argument(
        "--teacher-n-hidden", default=400, type=int,
        help="Number of hidden units")
    parser.add_argument(
        "--meta-discount", default=0.99, type=float, 
        help="Discount factor")

    # Env
    parser.add_argument(
        "--env-name", type=str, required=True,
        help="OpenAI gym environment name")
    parser.add_argument(
        "--ep-max-timesteps", type=int, required=True,
        help="Episode is terminated when max timestep is reached.")

    # Misc
    parser.add_argument(
        "--save-models", action="store_true", 
        help="Whether or not models are saved")
    parser.add_argument(
        "--render", action="store_true", 
        help="Whether or not render")
    parser.add_argument(
        "--prefix", default="", type=str,
        help="Prefix for tb_writer and logging")
    parser.add_argument(
        "--seed", default=0, type=int, 
        help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument(
        "--window-size", default=15, type=int, 
        help="Window size")

    args = parser.parse_args()

    # Set log name
    args.log_name = \
        "env::%s_seed::%s_tau::%s_n_eval::%s_start::%s_noise_std::%s_batch_size::%s_" \
        "student_done::%s_prefix::%s_log" % (
            args.env_name, str(args.seed), args.tau, args.n_eval, args.start_timesteps, 
            args.expl_noise, args.batch_size, str(args.student_done), args.prefix)

    main(args=args)
