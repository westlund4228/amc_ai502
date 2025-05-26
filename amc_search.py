# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

import os
import numpy as np
import argparse
from copy import deepcopy
import torch
torch.backends.cudnn.deterministic = True

from env.channel_pruning_env import ChannelPruningEnv
from lib.agent import DDPG
from lib.utils import get_output_folder

from sensitive_layer import SensitiveLayerFinder
from torch.utils.data import DataLoader
from lib.data import get_dataset

from tensorboardX import SummaryWriter
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='AMC search script')

    parser.add_argument('--job', default='train', type=str, help='support option: train/export')
    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
    # env
    parser.add_argument('--model', default='mobilenet', type=str, help='model to prune')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to use (cifar/imagenet)')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    parser.add_argument('--preserve_ratio', default=0.5, type=float, help='preserve ratio of the model')
    parser.add_argument('--lbound', default=0.2, type=float, help='minimum preserve ratio')
    parser.add_argument('--rbound', default=1., type=float, help='maximum preserve ratio')
    parser.add_argument('--reward', default='acc_reward', type=str, help='Setting the reward')
    parser.add_argument('--acc_metric', default='acc5', type=str, help='use acc1 or acc5')
    parser.add_argument('--use_real_val', dest='use_real_val', action='store_true')
    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')
    # parser.add_argument('--pruning_method', default='cp', type=str,
    #                     help='method to prune (fg/cp for fine-grained and channel pruning)')
    # only for channel pruning
    parser.add_argument('--n_calibration_batches', default=60, type=int,
                        help='n_calibration_batches')
    parser.add_argument('--n_points_per_layer', default=10, type=int,
                        help='method to prune (fg/cp for fine-grained and channel pruning)')
    parser.add_argument('--channel_round', default=8, type=int, help='Round channel to multiple of channel_round')
    # ddpg
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for actor')
    parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
    parser.add_argument('--warmup', default=100, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=1., type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=100, type=int, help='memory size for each layer')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    # noise (truncated normal distribution)
    parser.add_argument('--init_delta', default=0.5, type=float,
                        help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.95, type=float,
                        help='delta decay during exploration')
    # training
    parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
    parser.add_argument('--output', default='./logs', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_episode', default=800, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
    parser.add_argument('--n_worker', default=16, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=50, type=int, help='number of data batch size')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # export
    parser.add_argument('--ratios', default=None, type=str, help='ratios for pruning')
    parser.add_argument('--channels', default=None, type=str, help='channels after pruning')
    parser.add_argument('--export_path', default=None, type=str, help='path for exporting models')
    parser.add_argument('--use_new_input', dest='use_new_input', action='store_true', help='use new input feature')
    parser.add_argument('--cfg_path', default=None, type=str,
                    help='Path to search result folder (should contain channels.txt)')
    parser.add_argument('--policy_path', default=None, type=str, help='Path to the policy file')

    return parser.parse_args()


def get_model_and_checkpoint(model, dataset, checkpoint_path, n_gpu=1):
    print('=> Building model..')
    if model == 'mobilenet_cifar':
        from models.mobilenet_cifar import MobileNet_CIFAR
        net = MobileNet_CIFAR(n_class=10)
    elif model == 'resnet_cifar':
        from models.resnet_cifar import ResNet, BasicBlock
        net = ResNet(BasicBlock, [2, 2, 2, 2], n_class=10)
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported for model loading")

    print(f"=> Loading checkpoint from {checkpoint_path}...")
    sd = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu') # Add map_location for flexibility
    if 'state_dict' in sd:  # a checkpoint but not a state_dict
        sd = sd['state_dict']
    # Remove 'module.' prefix if trained with DataParallel
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    # Load state dict, allow partial loading if strict=False might be needed depending on checkpoint origin
    net.load_state_dict(sd) # Remove strict=False initially unless errors occur
    net = net.cuda()
    if n_gpu > 1:
        net = torch.nn.DataParallel(net, list(range(n_gpu)))

    return net, deepcopy(net.state_dict())


def save_resume_state(agent, env, episode, output):
    state = {
        'agent_actor': agent.actor.state_dict(),
        'agent_critic': agent.critic.state_dict(),
        'agent_actor_target': agent.actor_target.state_dict(),
        'agent_critic_target': agent.critic_target.state_dict(),
        'agent_optim_actor': agent.actor_optim.state_dict(),
        'agent_optim_critic': agent.critic_optim.state_dict(),
        'agent_memory': agent.memory,
        'env_best_reward': env.best_reward,
        'env_best_strategy': env.best_strategy,
        'episode': episode,
    }
    with open(os.path.join(output, 'resume.pkl'), 'wb') as f:
        pickle.dump(state, f)

def load_resume_state(agent, env, output):
    with open(os.path.join(output, 'resume.pkl'), 'rb') as f:
        state = pickle.load(f)
    agent.actor.load_state_dict(state['agent_actor'])
    agent.critic.load_state_dict(state['agent_critic'])
    agent.actor_target.load_state_dict(state['agent_actor_target'])
    agent.critic_target.load_state_dict(state['agent_critic_target'])
    agent.actor_optim.load_state_dict(state['agent_optim_actor'])
    agent.critic_optim.load_state_dict(state['agent_optim_critic'])
    agent.memory = state['agent_memory']
    env.best_reward = state['env_best_reward']
    env.best_strategy = state['env_best_strategy']
    return state['episode']

def train(num_episode, agent, env, output, start_episode=0):
    agent.is_training = True
    initial_prune = 0.1
    initial_preserve = 1.0 - initial_prune
    noise_std0 = 0.2
    noise_decay = 0.99
    noise_std = noise_std0
    ft_epochs = getattr(args, 'ft_epochs', 1)
    step = episode = episode_steps = 0
    episode_reward = 0.
    best_acc = 0.0
    observation = None
    T = []
    episode = start_episode
    while episode < num_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)
            current_layer = 0
            clamped_actions = []

        # agent pick action ...
        if episode == 0:
            action = initial_preserve
        elif episode <= args.warmup:
            action = agent.random_action()
            # action = sample_from_truncated_normal_distribution(lower=0., upper=1., mu=env.preserve_ratio, sigma=0.5)
        else:
            action = agent.select_action(observation, episode=episode)

        action = float(action) + np.random.randn() * noise_std
        action = float(np.clip(action, 0.0, 1.0))
        noise_std *= noise_decay

        # env response with next_observation, reward, terminate_info
        # clamp prune ratio for any sensitive layer to 10%
        # clamp prune_rate = 1 - action  ≤ 0.1 → action ≥ 0.9
        p_action = float(action)
        #print(f"[DEBUG] layer {current_layer}'s action is {p_action:.3f}")

        if current_layer in env.sensitive_ids:
        	#print(f"[CLAMP] layer {current_layer} is sensitive, capping action {p_action:.3f}→0.9")
        	action = max(action, initial_preserve)

        W = sum(env.org_channels)
        w_t = env.org_channels[current_layer]
        W_reduced = sum(p * env.org_channels[i] for i, p in enumerate(clamped_actions))
        W_rest = sum(env.org_channels[i] for i in range(current_layer+1, env.n_prunable_layer))
        duty_preserve = (args.preserve_ratio * W - W_reduced - W_rest) / w_t
        action = max(action, float(max(duty_preserve, 0.0)))

        clamped_actions.append(action)

        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        current_layer += 1

        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # fix-length, never reach here
        # if max_episode_length and episode_steps >= max_episode_length - 1:
        #     done = True

        # [optional] save intermideate model
        if episode % int(num_episode / 3) == 0:
            agent.save_model(output)
            save_resume_state(agent, env, episode, output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode
            print('#{}: episode_reward:{:.4f} acc: {:.4f}, ratio: {:.4f}'.format(episode, episode_reward,
                                                                                 info['accuracy'],
                                                                                 info['compress_ratio']))
            text_writer.write(
                '#{}: episode_reward:{:.4f} acc: {:.4f}, ratio: {:.4f}\n'.format(episode, episode_reward,
                                                                                 info['accuracy'],
                                                                                 info['compress_ratio']))
            final_reward = T[-1][0]
            # print('final_reward: {}'.format(final_reward))
            # agent observe and update policy
            for r_t, s_t, s_t1, a_t, done in T:
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    agent.update_policy()

            #agent.memory.append(
            #    observation,
            #    agent.select_action(observation, episode=episode),
            #    0., False
            #)

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            tfwriter.add_scalar('reward/last', final_reward, episode)
            tfwriter.add_scalar('reward/best', env.best_reward, episode)
            tfwriter.add_scalar('info/accuracy', info['accuracy'], episode)
            tfwriter.add_scalar('info/compress_ratio', info['compress_ratio'], episode)
            tfwriter.add_text('info/best_policy', str(env.best_strategy), episode)
            # record the preserve rate for each layer
            for i, preserve_rate in enumerate(env.strategy):
                tfwriter.add_scalar('preserve_rate/{}'.format(i), preserve_rate, episode)

            text_writer.write('best reward: {}\n'.format(env.best_reward))
            text_writer.write('best policy: {}\n'.format(env.best_strategy))

            if info['accuracy'] > best_acc:
            	best_acc = info['accuracy']
            	env.best_strategy = clamped_actions.copy()
            	print(f"New best clamped policy: {env.best_strategy}")
    text_writer.close()
    # Always save best policy and channels to best_policy.txt after training
    if hasattr(env, 'best_strategy') and hasattr(env, 'org_channels'):
        best_ratios = [float(r) for r in env.best_strategy]
        org_channels = env.org_channels
        pruned_channels = [max(1, int(round(r * c))) for r, c in zip(best_ratios, org_channels)]
        policy_path = os.path.join(output, "best_policy.txt")
        with open(policy_path, "w") as f:
            f.write("policy," + ",".join([str(float(t)) for t in best_ratios]) + "\n")
            f.write("channels," + ",".join([str(int(c)) for c in pruned_channels]) + "\n")

def export_model(env, args):
    import os
    assert args.export_path is not None, 'Please provide a valid export path'
    env.set_export_path(args.export_path)

    print('=> Original model channels: {}'.format(env.org_channels))

    if args.ratios:
        ratios = [float(r) for r in args.ratios.split(',')]
        assert len(ratios) == len(env.org_channels)
        channels = [int(r * c) for r, c in zip(ratios, env.org_channels)]

    elif args.channels:
        channels = [int(r) for r in args.channels.split(',')]
        assert len(channels) == len(env.org_channels)
        ratios = [c2 / c1 for c2, c1 in zip(channels, env.org_channels)]
    elif args.policy_path:
        # Read policy and channels from txt file
        with open(args.policy_path, "r") as f:
            lines = f.readlines()
        policy_line = [line for line in lines if line.startswith("policy,")][0]
        channels_line = [line for line in lines if line.startswith("channels,")][0]
        ratios = [float(x) for x in policy_line.strip().split(",")[1:]]
        channels = [int(x) for x in channels_line.strip().split(",")[1:]]
        assert len(ratios) == len(env.org_channels)
        assert len(channels) == len(env.org_channels)
    else:
        raise ValueError('Must provide one of --ratios, --channels, or --policy_path')

    print('=> Pruning with ratios: {}'.format(ratios))
    print('=> Channels after pruning: {}'.format(channels))

    for r in ratios:
        env.step(r)

    return


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    model, checkpoint = get_model_and_checkpoint(args.model, args.dataset, checkpoint_path=args.ckpt_path,
                                                 n_gpu=args.n_gpu)

    _, calib_loader, _ = get_dataset(args.dataset,
	                                 batch_size=args.data_bsize,
	                                 n_worker=args.n_worker,
	                                 data_root=args.data_root)
    calib_loader = DataLoader(calib_loader.dataset,
	                          batch_size=args.data_bsize,
	                          shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    finder = SensitiveLayerFinder(model, calib_loader, device)
    rank_exps = finder.compute_rank_expectations()
    sensitive_layers = finder.identify_sensitive_layers(rank_exps)
    print(f"Identified sensitive layers: {sensitive_layers}")

    env = ChannelPruningEnv(model, checkpoint, args.dataset,
                            preserve_ratio=1. if args.job == 'export' else args.preserve_ratio,
                            n_data_worker=args.n_worker, batch_size=args.data_bsize,
                            args=args, export_model=args.job == 'export', use_new_input=args.use_new_input)
    env.sensitive_layers = sensitive_layers

    sensitive_ids = []
    for idx, name in enumerate(env.layer_names):
    	if name in sensitive_layers:
    		sensitive_ids.append(idx)
    print("Sensitive layer indices:", sensitive_ids)
    env.sensitive_ids = sensitive_ids

    if args.job == 'train':
        if args.resume != 'default':
            args.output = args.resume
        else:
            base_folder_name = '{}_{}_r{}_search'.format(args.model, args.dataset, args.preserve_ratio)
            if args.suffix is not None:
                base_folder_name = base_folder_name + '_' + args.suffix
            args.output = get_output_folder(args.output, base_folder_name)
        print('=> Saving logs to {}'.format(args.output))
        tfwriter = SummaryWriter(logdir=args.output)
        log_mode = 'a' if args.resume != 'default' else 'w'
        text_writer = open(os.path.join(args.output, 'log.txt'), log_mode)
        print('=> Output path: {}...'.format(args.output))

        nb_states = env.layer_embedding.shape[1]
        nb_actions = 1  # just 1 action here

        args.rmsize = args.rmsize * env.n_prunable_layer  # for each layer
        print('** Actual replay buffer size: {}'.format(args.rmsize))

        agent = DDPG(nb_states, nb_actions, args)
        start_episode = 0
        if args.resume != 'default':
            # resume from checkpoint
            try:
                start_episode = load_resume_state(agent, env, args.output)
                print(f"=> Resumed from episode {start_episode}")
            except Exception as e:
                print(f"Resume failed: {e}")
                start_episode = 0
        train(args.train_episode, agent, env, args.output, start_episode=start_episode)
    elif args.job == 'export':
        export_model(env, args)
    else:
        raise RuntimeError('Undefined job {}'.format(args.job))
