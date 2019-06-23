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

    def add(self, task_id, episode):
        self.storage[task_id] = episode

    def sample(self, batch_size):
        task_id = np.random.randint(0, len(self.storage) - 1, size=1)[0]
        return (self.storage[task_id], self.storage[task_id + 1])

    # def sample(self, batch_size):
    #     task_id = np.random.randint(0, len(self.storage) - 1, size=batch_size)

    #     sampled_tasks = []
    #     for i in task_id: 
    #         sampled_tasks.append((self.storage[i], self.storage[i + 1]))

    #     return sampled_tasks

    def sample_for_modeler(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        items = [[] for _ in range(len(self.storage[0]))]

        for i in ind: 
            experience = self.storage[i]
            for i_item, item in enumerate(experience):
                items[i_item].append(np.array(item, copy=False))

        return [np.array(item) for item in items]
