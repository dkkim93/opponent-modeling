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

    def get_mse_loss(self, episodes_i, params, visualize=False):
        samples = episodes_i.observations
        labels = episodes_i.rewards
        predictions = self(samples / self.norm_coeff, params)
        mse_loss = nn.MSELoss()(predictions, labels)

        if visualize:
            self.visualize(samples, labels, predictions)
    
        return mse_loss

    def visualize(self, samples, labels, predictions):
        plt.scatter(samples.cpu().data.numpy(), labels.cpu().data.numpy(), label="Label")
        plt.scatter(samples.cpu().data.numpy(), predictions.cpu().data.numpy(), label="Predict")
        plt.legend()
        plt.show()

    def get_adapted_params(self, episodes_i):
        updated_params = None
        for _ in range(self.args.fast_batch_size):
            adaptation_loss = self.get_mse_loss(episodes_i, params=updated_params)
            updated_params = self.update_params(adaptation_loss, first_order=self.args.first_order)
            self.load_state_dict(updated_params)
    
        return updated_params

    def meta_train(self, task_id):
        if len(self.memory) < 2:
            return

        self.optimizer.zero_grad()
        old_params = parameters_to_vector(self.parameters())
        validation_losses = 0.
        for _ in range(self.args.meta_batch_size):
            # Sample episodes for task i and task (i+1)
            episodes_i, episodes_i_ = self.memory.sample()
        
            # Get adaptation loss based on episode
            adapted_params = self.get_adapted_params(episodes_i)
            validation_loss = self.get_mse_loss(episodes_i_, params=adapted_params, visualize=True)
            validation_losses += validation_loss
        
            # Revert back to the original weight
            vector_to_parameters(old_params, self.parameters())
        
        validation_losses.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        # Log validation loss
        validation_losses_detached = validation_losses.data.cpu().numpy().flatten()[0]
        self.log[self.args.log_name].info("Task ID {} :: Validation loss {:.5f}".format(task_id, validation_losses_detached))
        self.tb_writer.add_scalar("validation_loss", validation_losses_detached, task_id)
