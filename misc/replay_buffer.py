import numpy as np


class ReplayBuffer(object):
    """Replay buffer
    Ref: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    """
    def __init__(self, args):
        self.args = args
        self.storage = {}

    def __len__(self):
        return len(self.storage)

    def clear(self):
        self.storage.clear()
        assert len(self.storage) == 0

    def add(self, task_id, episodes):
        self.storage[task_id] = episodes

    def sample(self):
        # min_task_id = len(self.storage) - self.args.meta_batch_size
        # if min_task_id < 0:
        #     min_task_id = 0
        # task_id = np.random.randint(min_task_id, len(self.storage) - 1, size=1)[0]
        task_id = np.random.choice(np.arange(len(self.storage) - 1), p=self.exp_distribution())
        return (self.storage[task_id], self.storage[task_id + 1])

    def exp_distribution(self):
        exp_lambda = 0.1  # Higher value, steeper slope
        tasks = np.arange(len(self.storage) - 1) + 1
        exp_dist = exp_lambda * np.exp(-exp_lambda * tasks)
        exp_dist = np.flip(exp_dist / float(np.sum(exp_dist)))

        return exp_dist
