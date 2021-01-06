import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.tc_replay_buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg_c_ctc import MADDPG
from random import seed

USE_CUDA = True


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
        torch.cuda.set_device(torch.device("cuda:1"))
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    all_rewards = []
    all_penalties_1 = []
    all_penalties_2 = []

    b_t = 0.00001
    alpha_1 = 12
    alpha_2 = 0.2

    lb_t_1 = torch.from_numpy(np.random.rand(1)).float()
    lb_t_2 = torch.from_numpy(np.random.rand(1)).float()
    
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
    log_dir = str(run_dir / 'logs')
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

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
        episode_coll_penalties = []
        episode_dep_penalties = []
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
            ls_rollout_collectors_penalty = []
            ls_rollout_depositors_penalty = []

            collectors_penalty = []
            depositors_penalty = []
            lagrangian_rews = []

            for reward in rewards:
                all_agts_rews_pens = reward.squeeze(1)
                agt_rews_pens = all_agts_rews_pens[0]  # rewards and penalties are shared among agents
                rew = agt_rews_pens[0]
                coll_pen = agt_rews_pens[1]
                dep_pen = agt_rews_pens[2]
                lagrangian_rew = rew + ((lb_t_1 * coll_pen) + (lb_t_2 * dep_pen))
                # collect separately for plotting
                ls_rollout_rewards.append(rew)
                ls_rollout_collectors_penalty.append(coll_pen)
                ls_rollout_depositors_penalty.append(dep_pen)
                # collect for experience replay
                collectors_penalty.append(np.asarray([coll_pen] * config.n_agents))
                depositors_penalty.append(np.asarray([dep_pen] * config.n_agents))
                lagrangian_rews.append(np.asarray([lagrangian_rew] * config.n_agents))

            episode_rewards.append(np.mean(ls_rollout_rewards))
            episode_coll_penalties.append(np.mean(ls_rollout_collectors_penalty))
            episode_dep_penalties.append(np.mean(ls_rollout_depositors_penalty))

            lagrangian_rews = np.asarray(lagrangian_rews)
            collectors_penalty = np.asarray(collectors_penalty)
            depositors_penalty = np.asarray(depositors_penalty)

            replay_buffer.push(obs, agent_actions, lagrangian_rews, collectors_penalty, depositors_penalty, next_obs,
                               dones)

            obs = next_obs
            t += config.n_rollout_threads

            lb_t_ls_1 = []
            lb_t_ls_2 = []
            lb_t_ls_1_up = []
            lb_t_ls_2_up = []

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
                        penalty_helper_1, penalty_helper_2 = maddpg.update(sample, a_i, logger=logger)
                        lb_t_1_ = torch.max(torch.tensor(0.0),
                                            (lb_t_1 + ((penalty_helper_1.mean() - alpha_1).float() * b_t)))
                        lb_t_2_ = torch.max(torch.tensor(0.0),
                                            (lb_t_2 + ((penalty_helper_2.mean() - alpha_2).float() * b_t)))
                        lb_t_ls_1.append(lb_t_1_)
                        lb_t_ls_2.append(lb_t_2_)
                    lb_t_ls_1_up.append(torch.from_numpy(np.asarray(lb_t_ls_1)).mean())
                    lb_t_ls_2_up.append(torch.from_numpy(np.asarray(lb_t_ls_2)).mean())
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
                lb_t_1 = torch.from_numpy(np.asarray(lb_t_ls_1_up)).mean()
                lb_t_2 = torch.from_numpy(np.asarray(lb_t_ls_2_up)).mean()

        all_rewards.append(np.sum(episode_rewards))
        all_penalties_1.append(np.sum(episode_coll_penalties))
        all_penalties_2.append(np.sum(episode_dep_penalties))

        log_rew = np.mean(all_rewards[-1024:])
        log_penalty1 = np.mean(all_penalties_1[-1024:])
        log_penalty2 = np.mean(all_penalties_2[-1024:])
       
        logger.add_scalar("Mean cost over latest 1024 epi/Training:-", log_rew, ep_i)
        logger.add_scalar("Mean penalty_1 over latest 1024 epi/Training:-", log_penalty1, ep_i)
        logger.add_scalar("Mean penalty_2 over latest 1024 epi/Training:-", log_penalty2, ep_i)
        #logger.add_scalar('lbt1', lb_t_1, ep_i)
        #logger.add_scalar('lbt2', lb_t_2, ep_i)

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
    parser.add_argument("--env_id", default="fullobs_collect_treasure", help="Name of environment")
    parser.add_argument("--model_name", default="Treasure Collection",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=0, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=100000, type=int)
    parser.add_argument("--n_agents", default=8, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=100000, type=int)
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
