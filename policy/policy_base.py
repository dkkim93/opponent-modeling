import torch
import torch.nn as nn
from collections import OrderedDict


class PolicyBase(nn.Module):
    def __init__(self, sampler, log, tb_writer, args, name, policy_id):
        super(PolicyBase, self).__init__()
        self.sampler = sampler
        self.log = log
        self.tb_writer = tb_writer
        self.args = args
        self.name = name + str(policy_id)
        self.policy_id = policy_id

    def update_params(self, loss, first_order):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=not first_order)

        updated_params = OrderedDict()
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - self.args.adaptation_lr * grad

        self.load_state_dict(updated_params)

        return updated_params
