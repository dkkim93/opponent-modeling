import torch
import torch.nn as nn
import torch.nn.functional as F
from policy.policy_base import PolicyBase
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContinuousPolicy(PolicyBase):
    def __init__(self, sampler, log, tb_writer, args, name, policy_id):
        super(ContinuousPolicy, self).__init__(
            sampler=sampler, log=log, tb_writer=tb_writer, args=args,
            name=name, policy_id=policy_id)

        self.set_policy()
        self.norm_coeff = float(self.sampler.envs.observation_space.high[0])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.meta_lr)

    def set_policy(self):
        input_dim = self.sampler.envs.observation_space.shape[0]
        output_dim = self.sampler.envs.action_space.shape[0]

        setattr(self, self.name + "_l1", nn.Linear(input_dim, self.args.n_hidden))
        setattr(self, self.name + "_l2", nn.Linear(self.args.n_hidden, self.args.n_hidden))
        setattr(self, self.name + "_l3", nn.Linear(self.args.n_hidden, output_dim))
        self.to(device)

        self.log[self.args.log_name].info(
            "Set {} policy for {}\n{}".format(self.args.policy_type, self.name, self))

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        x = F.relu(F.linear(
            x, weight=params[self.name + "_l1.weight"],
            bias=params[self.name + "_l1.bias"]))
        x = F.relu(F.linear(
            x, weight=params[self.name + "_l2.weight"],
            bias=params[self.name + "_l2.bias"]))
        x = torch.sigmoid(F.linear(
            x, weight=params[self.name + "_l3.weight"],
            bias=params[self.name + "_l3.bias"]))

        return x
