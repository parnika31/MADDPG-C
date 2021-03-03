import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents_tc import DDPGAgent

MSELoss = torch.nn.MSELoss()


class MADDPG(object):

    def __init__(self, agent_init_params, nagents,
                 gamma=0.99, tau=0.001, lr=0.001, hidden_dim=128,
                 discrete_action=True):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to each critic

            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = nagents
        self.agents = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.pen_critic_dev = 'cpu'
        self.trgt_pen_critic_dev = 'cpu'
        self.pen2_critic_dev = 'cpu'
        self.trgt_pen2_critic_dev = 'cpu'
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                               observations)]

    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, all penalties, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, penalties_1, penalties_2, next_obs, dones = sample

        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()
        if self.discrete_action:
            all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                                zip(self.target_policies, next_obs)]
        else:
            all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                             next_obs)]
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)

        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))

        vf_in = torch.cat((*obs, *acs), dim=1)
        actual_value = curr_agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        # ----------------Update Penalty_1 Critic ----------------- #

        curr_agent.penalty_critic_optimizer.zero_grad()
        tar_pen_critic_q = curr_agent.penalty_tar_critic(trgt_vf_in)
        target_pen_value = (penalties_1[agent_i].view(-1, 1) + self.gamma *
                            tar_pen_critic_q *
                            (1 - dones[agent_i].view(-1, 1)))
        curr_pen_value = curr_agent.penalty_critic(vf_in)
        penalty_loss = MSELoss(curr_pen_value, target_pen_value.detach())
        penalty_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.penalty_critic.parameters(), 0.5)
        curr_agent.penalty_critic_optimizer.step()
        penalty_helper_1 = curr_pen_value.detach()

        # ----------------Update Penalty_2 Critic ----------------- #
        curr_agent.penalty2_critic_optimizer.zero_grad()
        tar_pen_2_critic_q = curr_agent.penalty2_tar_critic(trgt_vf_in)
        target_pen_2_value = (penalties_2[agent_i].view(-1, 1) + self.gamma *
                            tar_pen_2_critic_q *
                            (1 - dones[agent_i].view(-1, 1)))
        curr_pen_2_value = curr_agent.penalty2_critic(vf_in)
        penalty2_loss = MSELoss(curr_pen_2_value, target_pen_2_value.detach())
        penalty2_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.penalty2_critic.parameters(), 0.5)
        curr_agent.penalty2_critic_optimizer.step()
        penalty_helper_2 = curr_pen_2_value.detach()

        # ----------------Update Policy ----------------- #

        curr_agent.policy_optimizer.zero_grad()

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = []
        for i, pi, ob in zip(range(self.nagents), self.policies, obs):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
            elif self.discrete_action:
                all_pol_acs.append(onehot_from_logits(pi(ob)))
            else:
                all_pol_acs.append(pi(ob))
        vf_in = torch.cat((*obs, *all_pol_acs), dim=1)

        pol_loss = curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out ** 2).mean() * 1e-3
        (pol_loss).backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        if logger is not None:
                logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss,
                                'pen_1_loss': penalty_loss,
                                'pen_2_loss': penalty2_loss},
                               self.niter)
        return penalty_helper_1,penalty_helper_2

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
            soft_update(a.penalty_tar_critic, a.penalty_critic, self.tau)
            soft_update(a.penalty2_tar_critic, a.penalty2_critic, self.tau)

        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.penalty_critic.train()
            a.penalty2_critic.train()
            a.target_policy.train()
            a.target_critic.train()
            a.penalty_tar_critic.train()
            a.penalty2_tar_critic.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()

        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
            # print(self.pol_dev)
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.pen_critic_dev == device:
            for a in self.agents:
                a.penalty_critic = fn(a.penalty_critic)
            self.pen_critic_dev = device
        if not self.pen2_critic_dev == device:
            for a in self.agents:
                a.penalty2_critic = fn(a.penalty2_critic)
            self.pen2_critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device
        if not self.trgt_pen_critic_dev == device:
            for a in self.agents:
                a.penalty_tar_critic = fn(a.penalty_tar_critic)
            self.trgt_pen_critic_dev = device
        if not self.trgt_pen2_critic_dev == device:
            for a in self.agents:
                a.penalty2_tar_critic = fn(a.penalty2_tar_critic)
            self.trgt_pen2_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
            a.target_policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='gpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, nagents, gamma=0.99, tau=0.001, lr=0.001, hidden_dim=128):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        for acsp, obsp in zip(env.action_space, env.observation_space):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            num_in_critic = 0
            for oobsp in env.observation_space:
                num_in_critic += oobsp.shape[0]
            for oacsp in env.action_space:
                num_in_critic += get_shape(oacsp)

            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'nagents':nagents}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance
