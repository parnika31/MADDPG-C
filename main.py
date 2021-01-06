
import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg_c import MADDPG
from random import seed

USE_CUDA = True  # torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env

    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def run(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    seed(config.seed)
    if USE_CUDA:
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    all_rewards = []
    all_penalties = []
    b_t = 0.00001
    alpha = 3
    lb_t = torch.from_numpy(np.random.rand(1)).float()
    extra_cost = 1000

    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(str(log_dir))
    
    logger = SummaryWriter(str(log_dir))
    
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    
    maddpg = MADDPG.init_from_env(env, nagents=config.n_agents,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim, gamma=config.gamma)
    
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        maddpg.prep_rollouts(device='cpu')
        episode_rewards = []
        episode_penalties = []
        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(
            config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)

            ls_rollout_rewards = []
            ls_rollout_penalties = []
            penalties = []
            for reward in rewards:
                penalty = 0
                if (reward[0] - extra_cost) > 0:
                    for idx in range(len(reward)):
                        reward[idx] -= extra_cost
                    penalty = 1
                    ls_rollout_rewards.append(reward[0])
                    ls_rollout_penalties.append(penalty)
                    penalties.append(np.asarray([penalty] * config.n_agents))
                    for idx in range(len(reward)):
                        reward[idx] += lb_t * penalty
                else:
                    ls_rollout_rewards.append(reward[0])
                    ls_rollout_penalties.append(penalty)
                    penalties.append(np.asarray([penalty] * config.n_agents))

            episode_rewards.append(np.mean(ls_rollout_rewards))
            episode_penalties.append(np.mean(ls_rollout_penalties))
            penalties = np.asarray(penalties)

            replay_buffer.push(obs, agent_actions, rewards, penalties, next_obs, dones)

            obs = next_obs
            t += config.n_rollout_threads
            lb_t_ls = []
            lb_t_ls_up = []

            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')

                for u_i in range(config.num_updates):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        penalty_helper = maddpg.update(sample, a_i, logger=logger)
                        lb_t_ = torch.max(torch.tensor(0.0), (lb_t + ((penalty_helper.mean() - alpha).float() * b_t)))
                        lb_t_ls.append(lb_t_)

                    lb_t_ls_up.append(torch.from_numpy(np.asarray(lb_t_ls)).mean())
                    maddpg.update_all_targets()

                maddpg.prep_rollouts(device='cpu')
                lb_t = torch.from_numpy(np.asarray(lb_t_ls_up)).mean()
                
        all_rewards.append(np.sum(episode_rewards))
        all_penalties.append(np.sum(episode_penalties))

        log_rew = np.mean(all_rewards[-1024:])
        log_penalty1 = np.mean(all_penalties[-1024:])

        logger.add_scalar("Mean cost over latest 1024 epi/Training:-", log_rew, ep_i)
        logger.add_scalar("Mean penalty_1 over latest 1024 epi/Training:-", log_penalty1, ep_i)
        #logger.add_scalar('lbt', lb_t, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            maddpg.prep_rollouts(device='cpu')
            os.makedirs(str(run_dir / 'incremental'), exist_ok=True)
            maddpg.save(str(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1))))
            maddpg.save(str(run_dir / 'model.pt'))

    maddpg.save(str(run_dir / 'model.pt'))
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="simple_spread", help="Name of environment")
    parser.add_argument("--model_name", default="simple_spread",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=0, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=200000, type=int)
    parser.add_argument("--n_agents", default=5, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=50, type=int)
    parser.add_argument("--num_updates", default=2, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=200000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--discrete_action",
                        default='True')

    config = parser.parse_args()

    run(config)

