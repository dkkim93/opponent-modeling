import numpy as np


class ReplayBuffer(object):
    """Replay buffer
    Ref: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    """
    def __init__(self):
        self.storage = {}

    def __len__(self):
        return len(self.storage)

    def clear(self):
        self.storage.clear()
        assert len(self.storage) == 0

    def add(self, task_id, episodes):
        self.storage[task_id] = episodes

    def sample(self):
        task_id = np.random.randint(0, len(self.storage) - 1, size=1)[0]
        return (self.storage[task_id], self.storage[task_id + 1])
