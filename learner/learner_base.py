import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from misc.replay_buffer import ReplayBuffer


class LearnerBase(nn.Module):
    def __init__(self, policy, sampler, log, tb_writer, args, name, learner_id):
        super(LearnerBase, self).__init__()
        self.policy = policy
        self.sampler = sampler
        self.log = log
        self.tb_writer = tb_writer
        self.args = args
        self.name = name + str(learner_id)
        self.learner_id = learner_id
        self.memory = ReplayBuffer(args)

    def add_episodes(self, task_id, episodes):
        self.memory.add(task_id, episodes)

    def get_mse_loss(self, episodes_i, params):
        samples = episodes_i.observations
        labels = episodes_i.rewards
        predictions = self.policy(samples / self.norm_coeff, params)
        mse_loss = nn.MSELoss()(predictions, labels)

        return mse_loss, predictions

    def get_adapted_params(self, episodes_i):
        updated_params = None
        for _ in range(self.args.fast_batch_size):
            adaptation_loss, _ = self.get_mse_loss(episodes_i, params=updated_params)
            updated_params = self.policy.update_params(adaptation_loss, first_order=self.args.first_order)
            self.policy.load_state_dict(updated_params)
    
        return updated_params

    def visualize(self, episodes_i, episodes_i_, predictions_, task_id):
        samples = episodes_i.observations.cpu().data.numpy()
        labels = episodes_i.rewards.cpu().data.numpy()
        samples_ = episodes_i_.observations.cpu().data.numpy()
        labels_ = episodes_i_.rewards.cpu().data.numpy()
        predictions_ = predictions_.cpu().data.numpy()

        plt.scatter(samples, labels, label="Label")
        plt.scatter(samples_, labels_, label="Label_")
        plt.scatter(samples_, predictions_, label="Prediction_")
        plt.legend()
        plt.savefig("./logs/" + str(self.args.log_name) + "_" + str(task_id) + ".png", bbox_inches="tight")
        plt.close()

        np.save("./logs/" + str(self.args.log_name) + "_samples_" + str(task_id) + ".npy", samples)
        np.save("./logs/" + str(self.args.log_name) + "_labels_" + str(task_id) + ".npy", labels)
        np.save("./logs/" + str(self.args.log_name) + "_samples__" + str(task_id) + ".npy", samples_)
        np.save("./logs/" + str(self.args.log_name) + "_labels__" + str(task_id) + ".npy", labels_)
        np.save("./logs/" + str(self.args.log_name) + "_predictions__" + str(task_id) + ".npy", predictions_)

    def train(self, task_id):
        raise NotImplementedError("")

    def test(self, task_id, episodes_i_):
        raise NotImplementedError("")
