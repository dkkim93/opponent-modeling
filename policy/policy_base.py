import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from misc.replay_buffer import ReplayBuffer
from collections import OrderedDict
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters


class PolicyBase(nn.Module):
    def __init__(self, sampler, log, tb_writer, args, name, policy_id):
        super(PolicyBase, self).__init__()
        self.sampler = sampler
        self.log = log
        self.tb_writer = tb_writer
        self.args = args
        self.name = name + str(policy_id)
        self.policy_id = policy_id
        self.memory = ReplayBuffer()

    def update_params(self, loss, first_order):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters
        """
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=not first_order)

        updated_params = OrderedDict()
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - self.args.fast_lr * grad

        return updated_params

    def add_episodes(self, task_id, episodes):
        self.memory.add(task_id, episodes)

    def get_mse_loss(self, episodes_i, params):
        samples = episodes_i.observations
        labels = episodes_i.rewards
        predictions = self(samples / self.norm_coeff, params)
        mse_loss = nn.MSELoss()(predictions, labels)

        return mse_loss, predictions

    def get_adapted_params(self, episodes_i):
        updated_params = None
        for _ in range(self.args.fast_batch_size):
            adaptation_loss, _ = self.get_mse_loss(episodes_i, params=updated_params)
            updated_params = self.update_params(adaptation_loss, first_order=self.args.first_order)
            self.load_state_dict(updated_params)
    
        return updated_params

    def meta_train(self, task_id):
        if len(self.memory) < 2:
            return

        self.optimizer.zero_grad()
        old_params = parameters_to_vector(self.parameters())
        val_losses = 0.
        for i_meta_batch in range(self.args.meta_batch_size):
            if i_meta_batch == 0:
                # As online learning, ensure to include up-to-date task
                episodes_i = self.memory.storage[task_id - 1]
                episodes_i_ = self.memory.storage[task_id]
            else:
                # Sample episodes for task i and task (i+1)
                episodes_i, episodes_i_ = self.memory.sample()
        
            # Get adaptation loss based on episode
            adapted_params = self.get_adapted_params(episodes_i)
            val_loss, _ = self.get_mse_loss(episodes_i_, params=adapted_params)
            val_losses += val_loss
        
            # Revert back to the original weight
            vector_to_parameters(old_params, self.parameters())
        
        val_losses.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        # Log validation loss
        val_losses_detached = val_losses.data.cpu().numpy().flatten()[0]
        self.log[self.args.log_name].info("Task ID {} :: Validation loss {:.5f}".format(task_id, val_losses_detached))
        self.tb_writer.add_scalars("loss", {"val_loss": val_losses_detached}, task_id)

    def meta_test(self, task_id, episodes_i_):
        if task_id < 1:
            return

        old_params = parameters_to_vector(self.parameters())

        # Sample episodes for task i
        episodes_i = self.memory.storage[task_id - 1]
        
        # Get adaptation loss based on episode
        adapted_params = self.get_adapted_params(episodes_i)
        test_loss, predictions_ = self.get_mse_loss(episodes_i_, params=adapted_params)

        # Visualize
        self.visualize(episodes_i, episodes_i_, predictions_, task_id)
        
        # Revert back to the original weight
        vector_to_parameters(old_params, self.parameters())
        
        # Log test loss
        test_loss_detached = test_loss.data.cpu().numpy().flatten()[0]
        self.log[self.args.log_name].info("Task ID {} :: Test loss {:.5f}".format(task_id, test_loss_detached))
        self.tb_writer.add_scalars("loss", {"test_loss": test_loss_detached}, task_id)

    def visualize(self, episodes_i, episodes_i_, predictions_, task_id):
        samples = episodes_i.observations.cpu().data.numpy()
        labels = episodes_i.rewards.cpu().data.numpy()
        samples_ = episodes_i_.observations.cpu().data.numpy()
        labels_ = episodes_i_.rewards.cpu().data.numpy()

        plt.scatter(samples, labels, label="Label")
        plt.scatter(samples_, labels_, label="Label_")
        plt.scatter(samples_, predictions_.cpu().data.numpy(), label="Prediction_")
        plt.legend()
        plt.savefig("./logs/" + str(self.args.log_name) + "_" + str(task_id) + ".png", bbox_inches="tight")
        plt.close()
