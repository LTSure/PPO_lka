import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous


def evaluate_policy(args, env, agent, state_norm):
    times = 5
    evaluate_reward = 0
    for _ in range(times):
        s, info = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        episode_steps = 0
        while not done:
            episode_steps += 1
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, truncated, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
            if episode_steps >= args.max_episode_steps:
                done = True
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, env_name, number, seed):
    env = gym.make(env_name)
    env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env_evaluate.reset(seed=seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    # Build a tensorboard
    time = datetime.now()
    time_str = time.strftime("%m-%d-%H-%M")
    log_dir = 'runs/PPO_continuous/{}/{}_{}_{}_{}_seed_{}'.format(env_name, time_str, args.policy_dist, args.log_name, number, seed)
    writer = SummaryWriter(log_dir=log_dir)
    args.writer = writer
    log_file_path = os.path.join(log_dir, "logs.txt")


    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)


    
    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        s, info = env.reset()
        initial_s = s
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        episode_return = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, truncated, _ = env.step(action)
            episode_return += r
            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.


            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False
            
            if episode_steps >= args.max_episode_steps - 1:
                done = True          

            # Take the 'action'，but store the original 'a'（especially for Beta）
            current_traj_return = episode_return
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done, current_traj_return, initial_s)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                
                evaluation = "evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward)
                print(evaluation)
                with open(log_file_path,'a+') as f:
                    f.write("{}:\t{}\n".format(time_str, evaluation))
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save(os.path.join(log_dir, 'PPO_continuous_{}_{}_{}_{}.npy'.format(args.policy_dist, env_name, number, seed)), np.array(evaluate_rewards))




if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e4, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20000, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--device", type=str, default="cuda", help="Name of the environment")
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v4", help="Name of the environment")
    parser.add_argument("--random_seed", type=float, default=50, help="Random Seed")
    parser.add_argument("--foresight_eta", type=float, default=0, help="Extent to foresee the distribution")
    parser.add_argument("--log_name", type=str, default="0_scale_nodecay", help="Extent to foresee the distribution")


    args = parser.parse_args()
    env_name = args.env_name
    seed = args.random_seed


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    args.device = device
    
    main(args, env_name=env_name, number=1, seed=seed)
