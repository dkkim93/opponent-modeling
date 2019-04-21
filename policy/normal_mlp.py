import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.utils import normal_dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(torch.nn.Module):
    def __init__(self, actor_input_dim, actor_output_dim, n_hidden, max_action, name):
        super(Actor, self).__init__()

        setattr(self, name + "_l1", nn.Linear(actor_input_dim, n_hidden))
        setattr(self, name + "_l2", nn.Linear(n_hidden, n_hidden))
        setattr(self, name + "_l3_mu", nn.Linear(n_hidden, actor_output_dim))
        setattr(self, name + "_l3_std", nn.Linear(n_hidden, actor_output_dim))

        self.max_action = max_action
        self.name = name

        assert max_action >= 0., "Max action should be positive or zero"

    def forward(self, x):
        x = F.relu(getattr(self, self.name + "_l1")(x))
        x = F.relu(getattr(self, self.name + "_l2")(x))

        mu = F.softsign(getattr(self, self.name + "_l3_mu")(x))
        mu = torch.clamp(mu, -self.max_action, +self.max_action)

        std = getattr(self, self.name + "_l3_std")(x)
        std = F.softplus(std) + 1e-5

        return mu, std


class NormalMLP(object):
    def __init__(self, actor_input_dim, actor_output_dim, n_hidden, max_action, name, args):
        self.actor = Actor(actor_input_dim, actor_output_dim, n_hidden, max_action, name=name + "_actor").to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.max_action = max_action
        self.name = name

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        mu, std = self.actor(state)

        return mu.cpu().data.numpy().flatten(), std.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size):
        debug = {}
        debug["actor_loss"] = 0.

        # Sample opponent state and action
        items = replay_buffer.sample_for_modeler(batch_size)
        assert len(items) == 2, "items should contain state and actor"

        state = torch.FloatTensor(items[0]).to(device)
        opponent_action = torch.FloatTensor(items[1]).to(device)

        # Select action by modeler
        modeler_mu, modeler_std = self.actor(state)

        # Compute log prob
        prob = normal_dist(opponent_action, modeler_mu, modeler_std, device=device)
        log_prob = (prob + 1e-5).log()

        # Compute loss
        actor_loss = -log_prob.mean()

        # Optimize modeler
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        debug["actor_loss"] += actor_loss.cpu().data.numpy().flatten()

        return debug
