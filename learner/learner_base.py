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

    def get_mse_loss(self, episodes_i, params, i_agent):
        samples = episodes_i.observations[:, :, i_agent]
        labels = episodes_i.rewards[:, :, i_agent]
        predictions = self.policy(samples / self.norm_coeff, params)
        mse_loss = nn.MSELoss()(predictions, labels)

        return mse_loss, predictions

    def get_adapted_params(self, episodes_i, i_agent):
        updated_params = None
        for _ in range(self.args.fast_batch_size):
            adaptation_loss, _ = self.get_mse_loss(episodes_i, params=updated_params, i_agent=i_agent)
            updated_params = self.policy.update_params(adaptation_loss, first_order=self.args.first_order)
            self.policy.load_state_dict(updated_params)
    
        return updated_params

    def visualize(self, episodes_i, episodes_i_, predictions_, task_id):
        for i_agent in range(self.args.n_agent):
            sample = episodes_i.observations[:, :, i_agent].cpu().data.numpy()
            label = episodes_i.rewards[:, :, i_agent].cpu().data.numpy()
            sample_ = episodes_i_.observations[:, :, i_agent].cpu().data.numpy()
            label_ = episodes_i_.rewards[:, :, i_agent].cpu().data.numpy()
            prediction_ = predictions_[i_agent].cpu().data.numpy()

            # plt.scatter(sample, label, label="Label" + str(i_agent))
            plt.scatter(sample_, label_, label="Label_" + str(i_agent))
            plt.scatter(sample_, prediction_, label="Prediction_" + str(i_agent))

            np.save("./logs/" + str(self.args.log_name) + "_samples_" + str(task_id) + "_" + str(i_agent) + ".npy", sample)
            np.save("./logs/" + str(self.args.log_name) + "_labels_" + str(task_id) + "_" + str(i_agent) + ".npy", label)
            np.save("./logs/" + str(self.args.log_name) + "_samples__" + str(task_id) + "_" + str(i_agent) + ".npy", sample_)
            np.save("./logs/" + str(self.args.log_name) + "_labels__" + str(task_id) + "_" + str(i_agent) + ".npy", label_)
            np.save(
                "./logs/" + str(self.args.log_name) + "_predictions__" + str(task_id) + 
                "_" + str(i_agent) + ".npy", prediction_)
        plt.legend()
        plt.savefig("./logs/" + str(self.args.log_name) + "_" + str(task_id) + ".png", bbox_inches="tight")
        plt.close()

    def train(self, task_id):
        raise NotImplementedError("")

    def test(self, task_id, episodes_i_):
        raise NotImplementedError("")
