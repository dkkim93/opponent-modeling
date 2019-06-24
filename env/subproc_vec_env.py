import numpy as np
import multiprocessing as mp
import gym
import queue as queue


class EnvWorker(mp.Process):
    def __init__(self, remote, env_fn, queue, lock, worker_id, args):
        super(EnvWorker, self).__init__()
        self.remote = remote
        self.env = env_fn()
        self.env._max_episode_steps = args.ep_max_timesteps
        self.queue = queue
        self.lock = lock
        self.task_id = None
        self.done = False
        self.id = worker_id

    def seed(self, seed):
        self.env.seed(seed)

    def empty_step(self):
        observation = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        reward, done = 0.0, True
        return observation, reward, done, {}

    def try_reset(self):
        with self.lock:
            try:
                self.task_id = self.queue.get(True)
                self.done = (self.task_id is None)
            except queue.Empty:
                self.done = True
        observation = (np.zeros(self.env.observation_space.shape, dtype=np.float32) if self.done else self.env.reset())
        return observation

    def run(self):
        while True:
            command, data = self.remote.recv()

            if command == 'step':
                observation, reward, done, info = (self.empty_step() if self.done else self.env.step(data))
                if done and (not self.done):
                    observation = self.try_reset()
                self.remote.send((observation, reward, done, self.task_id, info))
            elif command == 'reset':
                observation = self.try_reset()
                self.remote.send((observation, self.task_id))
            elif command == 'reset_task':
                self.env.unwrapped.reset_task(data)
                self.remote.send(True)
            elif command == 'close':
                self.remote.close()
                break
            elif command == 'get_spaces':
                self.remote.send(
                    (self.env.observation_space, self.env.action_space))
            else:
                raise NotImplementedError()


class SubprocVecEnv(gym.Env):
    def __init__(self, envs, queue, args):
        self.lock = mp.Lock()
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in envs])
        worker_ids = list(range(len(envs)))
        self.workers = [
            EnvWorker(remote, env_fn, queue, self.lock, worker_id, args) 
            for (remote, env_fn, worker_id) in zip(self.work_remotes, envs, worker_ids)]
        for worker in self.workers:
            worker.daemon = True
            worker.start()
        for remote in self.work_remotes:
            remote.close()
        self.waiting = False
        self.closed = False

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def seed(self, seed):
        for worker in self.workers:
            worker.seed(seed + worker.id)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        observations, rewards, dones, task_ids, infos = zip(*results)
        return np.stack(observations), np.stack(rewards), np.stack(dones), task_ids, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        observations, task_ids = zip(*results)
        return np.stack(observations), task_ids

    def reset_task(self, tasks):
        for remote, task in zip(self.remotes, tasks):
            remote.send(('reset_task', task))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()
        self.closed = True
