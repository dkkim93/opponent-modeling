import torch
from learner.learner_base import LearnerBase
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters


class MetaLearner(LearnerBase):
    def __init__(self, policy, sampler, log, tb_writer, args, name, learner_id):
        super(MetaLearner, self).__init__(
            policy=policy, sampler=sampler, log=log, tb_writer=tb_writer, args=args,
            name=name, learner_id=learner_id)
    
        self.norm_coeff = float(self.sampler.envs.observation_space.high[0])

    def train(self, task_id):
        if len(self.memory) < 2:
            return

        self.policy.optimizer.zero_grad()
        old_params = parameters_to_vector(self.policy.parameters())
        val_losses = 0.
        for i_meta_batch in range(self.args.meta_batch_size):
            if i_meta_batch == 0:
                # As online learning, ensure to include up-to-date task
                episodes_i = self.memory.storage[task_id - 1]
                episodes_i_ = self.memory.storage[task_id]
            else:
                # Sample episodes for task i and task (i+1)
                episodes_i, episodes_i_ = self.memory.sample()
            
            for i_agent in range(self.args.n_agent):
                # Get adaptation loss based on episode
                adapted_params = self.get_adapted_params(episodes_i, i_agent)
                val_loss, _ = self.get_mse_loss(episodes_i_, params=adapted_params, i_agent=i_agent)
                val_losses += val_loss
            
                # Revert back to the original weight
                vector_to_parameters(old_params, self.policy.parameters())
        
        val_losses.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.policy.optimizer.step()

        # Log validation loss
        val_losses_detached = val_losses.data.cpu().numpy().flatten()[0]
        self.log[self.args.log_name].info("Task ID {} :: Validation loss {:.5f}".format(task_id, val_losses_detached))
        self.tb_writer.add_scalars("loss", {"val_loss": val_losses_detached}, task_id)

    def test(self, task_id, episodes_i_):
        if task_id < 1:
            return

        # Sample episodes for task i
        episodes_i = self.memory.storage[task_id - 1]
            
        old_params = parameters_to_vector(self.policy.parameters())
        predictions_ = []
        for i_agent in range(self.args.n_agent):
            # Get adaptation loss based on episode
            adapted_params = self.get_adapted_params(episodes_i, i_agent)
            test_loss, prediction_ = self.get_mse_loss(episodes_i_, params=adapted_params, i_agent=i_agent)
            predictions_.append(prediction_)
            
            # Revert back to the original weight
            vector_to_parameters(old_params, self.policy.parameters())

        # Visualize
        self.visualize(episodes_i, episodes_i_, predictions_, task_id)
        
        # Log test loss
        test_loss_detached = test_loss.data.cpu().numpy().flatten()[0]
        self.log[self.args.log_name].info("Task ID {} :: Test loss {:.5f}".format(task_id, test_loss_detached))
        self.tb_writer.add_scalars("loss", {"test_loss": test_loss_detached}, task_id)
