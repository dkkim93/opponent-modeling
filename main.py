import torch
import argparse
import os
import numpy as np
import torch.nn as nn
from misc.utils import set_log, set_policy
from tensorboardX import SummaryWriter
from misc.batch_sampler import BatchSampler
from misc.replay_buffer import ReplayBuffer
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
# from trainer.train_opponent import train_opponent
# from trainer.train_modeler import train_modeler


def get_loss(meta_policy, episode, params, visualize=False):
    samples = episode.observations
    labels = episode.rewards
    predictions = meta_policy(samples / 20., params)

    mse = nn.MSELoss()
    loss = mse(predictions, labels)

    if visualize:
        import matplotlib.pyplot as plt
        plt.scatter(samples.cpu().data.numpy().flatten(), labels.cpu().data.numpy().flatten(), label="label")
        plt.scatter(samples.cpu().data.numpy().flatten(), predictions.cpu().data.numpy().flatten(), label="pred")
        plt.show()

    return loss


def get_adapted_params(meta_policy, episode, args, visualize):
    old_params = parameters_to_vector(meta_policy.parameters())
    updated_params = None
    for _ in range(10):
        adaptation_loss = get_loss(meta_policy, episode, updated_params, visualize)
        updated_params = meta_policy.update_params(adaptation_loss, first_order=True)

    return old_params, updated_params


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
    memory = ReplayBuffer()

    task_id = 0
    while True:
        print("Task_id:", task_id)

        # Sample experiences from current task
        sampler.reset_task(task_id)
        episode = sampler.sample()

        # Add to memory
        memory.add(task_id, episode)

        # Train meta-parameter
        if len(memory) > 1:
            meta_policy.optimizer.zero_grad()
            validation_losses = 0.
            for _ in range(20):
                if _ == 0:
                    if task_id % 20 == 0:
                        visualize = True
                    else:
                        visualize = False
                else:
                    visualize = False

                episode, next_episode = memory.sample(args.meta_batch_size)

                # Get adaptation loss based on episode
                old_params, adapted_params = get_adapted_params(meta_policy, episode, args, False)
                validation_loss = get_loss(meta_policy, episode, adapted_params, visualize=visualize)
                validation_losses += validation_loss

                # Revert back to the original weight
                vector_to_parameters(old_params, meta_policy.parameters())

            validation_losses.backward()
            torch.nn.utils.clip_grad_norm_(meta_policy.parameters(), 0.5)
            print(validation_losses)
            meta_policy.optimizer.step()

        # Move to next task
        task_id += 1

        if task_id > 1000:
            import sys
            sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # # TD3 Algorithm
    # parser.add_argument(
    #     "--discount", type=float, default=0.99,
    #     help="Discount factor")
    # parser.add_argument(
    #     "--tau", type=float, default=0.01,
    #     help="Target network update rate")
    # parser.add_argument(
    #     "--start-timesteps", default=1e4, type=int, 
    #     help="How many time steps purely random policy is run for")
    # parser.add_argument(
    #     "--expl-noise", default=0.1, type=float, 
    #     help="Std of Gaussian exploration noise")
    # parser.add_argument(
    #     "--batch-size", default=50, type=int, 
    #     help="Batch size for both actor and critic")
    # parser.add_argument(
    #     "--policy-noise", default=0.2, type=float, 
    #     help="Noise added to target policy during critic update")
    # parser.add_argument(
    #     "--noise-clip", default=0.5, type=float, 
    #     help="Range to clip target policy noise")
    # parser.add_argument(
    #     "--policy-freq", default=2, type=int,
    #     help="Frequency of delayed policy updates")
    # parser.add_argument(
    #     "--actor-lr", default=0.0001, type=float,
    #     help="Learning rate for actor")
    # parser.add_argument(
    #     "--critic-lr", type=float, default=0.001,
    #     help="Learning rate for critic")
    # parser.add_argument(
    #     "--grad-clip", type=float, default=0.5, type=int, 
    #     help="Gradient clipping to prevent explosion")

    # # Opponent
    # parser.add_argument(
    #     "--train-opponent", action="store_true", 
    #     help="Whether or not models are saved")

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
    # parser.add_argument(
    #     "--train-meta", action="store_true", 
    #     help="Whether or not to train meta-learner")
    parser.add_argument(
        "--meta-batch-size", default=16, type=int, 
        help="Number of trajectory to collect from each task")
    parser.add_argument(
        "--adaptation-lr", default=10.0, type=float, 
        help="Adaptation learning rate")
    parser.add_argument(
        "--meta-lr", default=0.03, type=float, 
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
