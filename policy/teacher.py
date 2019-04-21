import numpy as np
import gym.spaces as spaces
from policy.td3 import TD3
from misc.replay_buffer import ReplayBuffer
from collections import OrderedDict

Q_VALUE_CLIP_MIN = -50.
Q_VALUE_CLIP_MAX = 0.


class Teacher(object):
    def __init__(self, env, tb_writer, log, args, name, i_agent):
        self.env = env
        self.tb_writer = tb_writer
        self.log = log
        self.args = args
        self.name = name + str(i_agent)
        self.i_agent = i_agent

        self.set_dim()
        self.set_policy()

        assert "teacher" in self.name

    def set_actor_input_dim(self):
        input_dim = 0

        # Add state
        input_dim += self.env.observation_space[0].shape[0]  # Teacher obs
        input_dim += self.env.observation_space[0].shape[0]  # Student obs
        if self.args.student_done:
            input_dim += 1  # +1 for remaining time
            input_dim += 1  # +1 for remaining time

        # Add action
        num_random_action = 15
        input_dim += 3 * num_random_action  # Teacher Q
        input_dim += 3 * num_random_action  # Student Q

        return input_dim

    def set_dim(self):
        self.actor_input_dim = self.set_actor_input_dim()
        self.actor_output_dim = self.env.action_space[0].shape[0] + 1
        self.critic_input_dim = 105 * 2 + 120 * 2 + 8  # NOTE Manual
        self.max_action = float(self.env.action_space[0].high[0])
        self.n_hidden = self.args.teacher_n_hidden

        self.log[self.args.log_name].info("[{}] Actor input dim: {}".format(
            self.name, self.actor_input_dim))
        self.log[self.args.log_name].info("[{}] Actor output dim: {}".format(
            self.name, self.actor_output_dim))
        self.log[self.args.log_name].info("[{}] Critic input dim: {}".format(
            self.name, self.critic_input_dim))
        self.log[self.args.log_name].info("[{}] Max action: {}".format(
            self.name, self.max_action))
        self.action_space = spaces.Box(low=-1, high=+1, shape=(self.actor_output_dim,), dtype=np.float32)

    def set_policy(self):
        self.policy = TD3(
            actor_input_dim=self.actor_input_dim,
            actor_output_dim=self.actor_output_dim,
            critic_input_dim=self.critic_input_dim,
            max_action=self.max_action,
            n_hidden=self.n_hidden,
            name=self.name,
            args=self.args,
            i_agent=self.i_agent)

        self.memory = ReplayBuffer()

    def select_stochastic_action(self, obs, total_step_count):
        if total_step_count < self.args.teacher_start_timesteps:
            action = self.action_space.sample()
            assert not np.isnan(action).any()
        else:
            action = self.policy.select_action(obs)
            assert not np.isnan(action).any()
            if self.args.expl_noise != 0:
                noise = np.random.normal(0, self.args.expl_noise, size=self.action_space.shape[0])
                action = (action + noise).clip(
                    self.action_space.low, self.action_space.high)

        for i_action in range(self.actor_output_dim):
            self.tb_writer.add_scalar(
                "action/teacher_advice" + str(self.i_agent) + "-" + str(i_action),
                action[i_action], total_step_count)

        return action

    def select_deterministic_action(self, obs):
        action = self.policy.select_action(obs)
        assert not np.isnan(action).any()

        return action

    def add_memory(self, obs, new_obs, action, reward, done):
        self.memory.add((obs, new_obs, action, reward, done))

    def clear_memory(self):
        self.memory.clear()

    def update_policy(self, agent_n, total_step_count):
        if len(self.memory) > self.args.ep_max_timesteps:
            debug = self.policy.centralized_train(
                agent_n=agent_n,
                replay_buffer=self.memory,
                iterations=self.args.ep_max_timesteps,
                batch_size=self.args.batch_size, 
                discount=self.args.meta_discount, 
                tau=self.args.tau, 
                policy_noise=self.args.policy_noise, 
                noise_clip=self.args.noise_clip, 
                policy_freq=self.args.policy_freq)

            self.tb_writer.add_scalars("loss/" + str(self.name), {
                "actor_loss": debug["actor_loss"],
                "critic_loss": debug["critic_loss"]}, total_step_count)

    def fix_name(self, weight):
        weight_fixed = OrderedDict()
        for k, v in weight.items():
            name_fixed = self.name
            for i_name, name in enumerate(k.split("_")):
                if i_name > 0:
                    name_fixed += "_" + name
            weight_fixed[name_fixed] = v

        return weight_fixed

    def reset(self):
        self.log[self.args.log_name].info("[{}] Reset".format(self.name))
        self.set_policy()

    def save_weight(self, filename, directory):
        self.log[self.args.log_name].info("[{}] Saved weight".format(self.name))
        self.policy.save(filename, directory)

    def load_weight(self, filename, directory):
        self.log[self.args.log_name].info("[{}] Loaded weight".format(self.name))
        self.policy.load(filename, directory)

    def set_eval_mode(self):
        self.log[self.args.log_name].info("[{}] Set eval mode".format(self.name))

        self.policy.actor.eval()
        self.policy.actor_target.eval()
        self.policy.critic.eval()
        self.policy.critic_target.eval()
