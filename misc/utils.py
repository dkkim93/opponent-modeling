import logging
import numpy as np
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv


class ReplayBuffer(object):
    """
        Simple replay buffer
        Ref: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
        NOTE Don't use this class. Use replay_buffer.py. Temporary here for saved weights
    """
    def __init__(self):
        self.storage = []

    def __len__(self):
        return len(self.storage)

    def clear(self):
        self.storage.clear()
        assert len(self.storage) == 0

    def add(self, data):
        # Expects tuples of (state, next_state, action, reward, done)
        if len(self.storage) > 1e6:
            self.storage.pop(0)
        self.storage.append(data)

    def sync(self, memory):
        self.clear()
        for exp in memory.storage:
            self.storage.append(exp)

        assert len(memory) == len(self.storage)

    def update_teacher_reward(self, update_type, reward, done=None):
        if update_type == "last_exp_only":
            exp = self.storage.pop(-1)

            if done is None:
                self.storage.append((
                    exp[0], 
                    exp[1], 
                    exp[2], 
                    reward, 
                    exp[4]))
            else:
                raise NotImplementedError()
                self.storage.append((
                    exp[0], 
                    exp[1], 
                    exp[2], 
                    [reward, reward], 
                    [done, done]))

        elif update_type == "last_traj_only":
            raise NotImplementedError()
        else:
            raise ValueError()

    def sample(self, batch_size=100):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind: 
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

    def centralized_sample(self, batch_size=100, n_agent=None):
        # NOTE Order is agent 0 and 1
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x_n, y_n, u_n, r_n, d_n = [], [], [], [], []

        for i_agent in range(n_agent):
            x, y, u, r, d = [], [], [], [], []

            for i in ind: 
                X, Y, U, R, D = self.storage[i]
                assert len(X) == n_agent

                x.append(np.array(X[i_agent], copy=False))
                y.append(np.array(Y[i_agent], copy=False))
                u.append(np.array(U[i_agent], copy=False))
                r.append(np.array(R[i_agent], copy=False))
                d.append(np.array(D[i_agent], copy=False))

            assert len(x) == batch_size
            x, y, u, r, d = np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

            x_n.append(x)
            y_n.append(y)
            u_n.append(u)
            r_n.append(r)
            d_n.append(d)

        assert len(x_n) == n_agent

        return x_n, y_n, u_n, r_n, d_n


def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_log(args):
    log = {}                                                                                                                                        
    set_logger(
        logger_name=args.log_name, 
        log_file=r'{0}{1}'.format("./logs/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    # Log arguments
    for (name, value) in vars(args).items():
        log[args.log_name].info("{}: {}".format(name, value))

    return log


def make_env(args):
    # Modified from: https://github.com/openai/maddpg/blob/master/experiments/train.py
    scenario = scenarios.load(args.env_name + ".py").Scenario()
    world = scenario.make_world(mode=2)
    done_callback = None

    env = MultiAgentEnv(
        world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        done_callback=done_callback)

    return env


def normalize(value, min_value, max_value):
    assert min_value < max_value
    return 2. * (value - min_value) / float(max_value - min_value) - 1.
