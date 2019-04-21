import copy
import numpy as np
# from misc.train_utils import *

total_timesteps = 0
total_eps = 0


def eval_progress(modeler, env, log, tb_writer, args, log_result):
    if total_eps % 3 == 0:
        eval_reward = 0.
        n_eval = 10

        for i_eval in range(n_eval):
            env_obs_n = env.reset()
            ep_timesteps = 0.
            terminal = False

            while True:
                if total_eps % 100 == 0:
                    env.render()

                modeler_obs_n = env_obs_n
                modeler_action = modeler.select_deterministic_action(np.array(modeler_obs_n[0]))

                new_env_obs_n, reward_n, done_n, _ = env.step(copy.deepcopy([modeler_action]))
                terminal = True if ep_timesteps + 1 == args.ep_max_timesteps else False

                # For next timestep
                env_obs_n = new_env_obs_n
                eval_reward += reward_n[0]
                ep_timesteps += 1

                if terminal:
                    break

        # Log result
        eval_reward /= float(n_eval)
        log[args.log_name].info("Evaluation Reward {:.5f} at episode {}".format(eval_reward, total_eps))
        tb_writer.add_scalar("reward/eval_reward", eval_reward, total_eps)


def collect_one_traj(modeler, opponent_n, env, log, args, tb_writer):
    global total_timesteps, total_eps

    ep_reward = 0.
    ep_timesteps = 0
    env_obs_n = env.reset()

    while True:
        # opponent selects its action
        opponent_obs_n = env_obs_n
        opponent_action_n = []
        for opponent, opponent_obs in zip(opponent_n, opponent_obs_n):
            opponent_action = opponent.select_deterministic_action(obs=np.array(opponent_obs))
            opponent_action_n.append(opponent_action)

        # Perform action
        new_env_obs_n, env_reward_n, env_done_n, _ = env.step(copy.deepcopy(opponent_action_n))
        terminal = True if ep_timesteps + 1 == args.ep_max_timesteps else False

        # Add to modeler memory
        modeler.add_memory(
            obs=opponent_obs_n[0],
            action=opponent_action_n[0])

        # For next timestep
        env_obs_n = new_env_obs_n
        ep_timesteps += 1
        total_timesteps += 1
        ep_reward += env_reward_n[0]

        if terminal: 
            total_eps += 1
            
            return ep_reward


def train_modeler(modeler, opponent_n, env, log, tb_writer, args):
    while True:
        eval_progress(
            modeler, env=env, log=log,
            tb_writer=tb_writer, args=args, log_result=True)

        collect_one_traj(
            modeler=modeler, opponent_n=opponent_n, env=env, 
            log=log, args=args, tb_writer=tb_writer)

        debug = modeler.update_policy(total_timesteps)
        tb_writer.add_scalar("loss/modeler_loss", debug["actor_loss"], total_timesteps)
