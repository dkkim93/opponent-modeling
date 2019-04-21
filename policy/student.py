import torch
import numpy as np
from policy.td3 import TD3
from misc.utils import ReplayBuffer
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Student(object):
    def __init__(self, env, log, args, name, i_agent):
        self.env = env
        self.log = log
        self.args = args
        self.name = name + str(i_agent)
        self.i_agent = i_agent

        self.set_dim()
        self.set_policy()

        self.actor_loss_n = []
        self.critic_loss_n = []

        assert "student" in self.name

    def set_dim(self):
        self.actor_input_dim = self.env.observation_space[0].shape[0]
        if self.args.student_done:
            self.actor_input_dim += 1
        self.actor_output_dim = self.env.action_space[0].shape[0] 
        self.critic_input_dim = (self.actor_input_dim + self.actor_output_dim) * self.args.n_student
        self.max_action = float(self.env.action_space[0].high[0])

        self.log[self.args.log_name].info("[{}] Actor input dim: {}".format(
            self.name, self.actor_input_dim))
        self.log[self.args.log_name].info("[{}] Actor output dim: {}".format(
            self.name, self.actor_output_dim))
        self.log[self.args.log_name].info("[{}] Critic input dim: {}".format(
            self.name, self.critic_input_dim))
        self.log[self.args.log_name].info("[{}] Max action: {}".format(
            self.name, self.max_action))

    def set_policy(self):
        self.policy = TD3(
            actor_input_dim=self.actor_input_dim,
            actor_output_dim=self.actor_output_dim,
            critic_input_dim=self.critic_input_dim,
            n_hidden=self.args.student_n_hidden,
            max_action=self.max_action,
            name=self.name,
            args=self.args,
            i_agent=self.i_agent)

        self.memory = ReplayBuffer()

    def select_stochastic_action(self, obs, total_timesteps):
        if total_timesteps < self.args.start_timesteps:
            action = self.env.action_space[0].sample()
            assert not np.isnan(action).any()
        else:
            action = self.policy.select_action(obs)
            assert not np.isnan(action).any()
            if self.args.expl_noise != 0:
                noise = np.random.normal(0, self.args.expl_noise, size=self.env.action_space[0].shape[0])
                action = (action + noise).clip(
                    self.env.action_space[0].low, self.env.action_space[0].high)

        return action

    def select_deterministic_action(self, obs):
        action = self.policy.select_action(obs)
        assert not np.isnan(action).any()

        return action

    def add_memory(self, obs, new_obs, action, reward, done):
        self.memory.add((obs, new_obs, action, reward, done))

    def clear_memory(self):
        self.memory.clear()

    def update_policy(self, student_n, total_timesteps):
        if total_timesteps % 2 == 0:
            policy_update = True
        else:
            policy_update = False

        debug = self.policy.centralized_task_level_train(
            agent_n=student_n,
            replay_buffer=self.memory,
            iterations=1,
            batch_size=self.args.batch_size, 
            discount=self.args.discount, 
            tau=self.args.tau, 
            policy_noise=self.args.policy_noise, 
            noise_clip=self.args.noise_clip, 
            policy_update=policy_update)

        return debug

    def fix_name(self, weight):
        weight_fixed = OrderedDict()
        for k, v in weight.items():
            name_fixed = self.name
            for i_name, name in enumerate(k.split("_")):
                if i_name > 0:
                    name_fixed += "_" + name
            weight_fixed[name_fixed] = v

        return weight_fixed

    def sync(self, target_agent):
        self.log[self.args.log_name].info("[{}] Synced weight".format(self.name))

        actor = self.fix_name(target_agent.policy.actor.state_dict())
        self.policy.actor.load_state_dict(actor)

        actor_target = self.fix_name(target_agent.policy.actor_target.state_dict())
        self.policy.actor_target.load_state_dict(actor_target)

        critic = self.fix_name(target_agent.policy.critic.state_dict())
        self.policy.critic.load_state_dict(critic)

        critic_target = self.fix_name(target_agent.policy.critic_target.state_dict())
        self.policy.critic_target.load_state_dict(critic_target)

        self.policy.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=self.args.actor_lr)
        self.policy.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=self.args.critic_lr)

    def get_q_value(self, obs, action):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(device)

        return self.policy.critic.Q1(obs, action).cpu().data.numpy().flatten()

    def reset(self):
        self.log[self.args.log_name].info("[{}] Reset".format(self.name))
        self.set_policy()
        self.actor_loss_n = []
        self.critic_loss_n = []

    def save_weight(self, filename, directory):
        self.log[self.args.log_name].info("[{}] Saved weight".format(self.name))
        self.policy.save(filename, directory)

    def load_weight(self, filename, directory):
        self.log[self.args.log_name].info("[{}] Loaded weight".format(self.name))
        self.policy.load(filename, directory)

    def load_model(self, filename, directory):
        if self.args.load_student_memory:
            self.log[self.args.log_name].info("[{}] Loaded memory".format(self.name))
            import pickle

            with open(directory + "/" + filename + ".pkl", "rb") as input_file:
                saved_model = pickle.load(input_file)

            self.memory.sync(saved_model["memory"])

        self.load_weight(filename, directory)

    def set_eval_mode(self):
        self.log[self.args.log_name].info("[{}] Set eval mode".format(self.name))

        self.policy.actor.eval()
        self.policy.actor_target.eval()
        self.policy.critic.eval()
        self.policy.critic_target.eval()

    def save_model(self, avg_eval_reward, total_ep_count):
        import pickle

        def save_pickle(obj, filename):
            with open(filename, "wb") as output:
                pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        # Filename by converting it to percentage
        filename = \
            self.name + \
            "_reward" + "{:.3f}".format(avg_eval_reward) + \
            "_seed" + str(self.args.seed) + \
            "_ep" + str(total_ep_count)

        # Save loss history & memory
        snipshot = {}
        snipshot["actor_loss_n"] = self.actor_loss_n
        snipshot["critic_loss_n"] = self.critic_loss_n
        snipshot["memory"] = self.memory

        save_pickle(
            obj=snipshot,
            filename=filename + ".pkl")

        # Save weight
        self.save_weight(
            filename=filename,
            directory="./pytorch_models")
