from utils import evaluate_policy, str2bool
from datetime import datetime
from DQN import DQN_agent
import gymnasium as gym
import os, shutil
import argparse
import torch


'''Hyperparameter Setting'''
# bash 输入 范例
# python Rainbow_main.py &
# python Rainbow_main.py --duel &
# python Rainbow_main.py --double &
# python Rainbow_main.py --double --duel &

parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')  # OK
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')  # OK env_index
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')  # OK summary_writer
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')  # OK render_human
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')  # OK
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')  # OK

parser.add_argument('--seed', type=int, default=0, help='random seed')  # OK
parser.add_argument('--Max_train_steps', type=int, default=int(1e6), help='Max training steps')  # OK
parser.add_argument('--save_interval', type=int, default=int(50e3), help='Model saving interval, in steps.')  # OK
parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')  # OK
parser.add_argument('--random_steps', type=int, default=int(3e3), help='steps for random policy to explore')  # OK
parser.add_argument('--update_every', type=int, default=50, help='training frequency')  # OK

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')  # OK
parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')  # OK
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')  # OK
parser.add_argument('--batch_size', type=int, default=256, help='length of sliced trajectory')  # OK
parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')  # OK eps_threshold; exploration noise 为了探索而加入的随机性
parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')  # OK
parser.add_argument('--Double', type=str2bool, default=True, help='Whether to use Double Q-learning')  # OK
parser.add_argument('--Duel', type=str2bool, default=True, help='Whether to use Duel networks')  # OK
opt = parser.parse_args()  # 解析传入的所有参数, 并返回 argparse.Namespace 类型的一个对象, 里面保存着所有参数值, 用opt.XXX 调用, 并不是dict
opt.dvc = torch.device(opt.dvc)
print(opt)


def main():
    EnvName = ['CartPole-v1','LunarLander-v2']
    BriefEnvName = ['CPV1', 'LLdV2']
    env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
    eval_env = gym.make(EnvName[opt.EnvIdex])
    opt.state_dim = env.observation_space.shape[0]  # 4
    opt.action_dim = env.action_space.n  # 2
    opt.max_e_steps = env._max_episode_steps  # 每个 episode 的最大步数 500

    # Algorithm Setting
    if opt.Duel: algo_name = 'Duel'
    else: algo_name = ''
    if opt.Double: algo_name += 'DDQN'
    else: algo_name += 'DQN'

    # Seed Everything, 为了 reproducibility 复现, 相同 seed 生成相同的随机数列
    env_seed = opt.seed  # env.reset(seed=env_seed); env_seed += 1 保证每次跑程序开始一样, 但是之后每个 episode 不一样, 不然 overfit 过拟合
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁止自选最快算法
    print("Random Seed: {}".format(opt.seed))

    print('Algorithm:',algo_name,'  Env:',BriefEnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,
          '  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps, '\n')

    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}-{}_S{}_'.format(algo_name,BriefEnvName[opt.EnvIdex],opt.seed) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    #Build model and replay buffer
    if not os.path.exists('model'): os.mkdir('model')
    agent = DQN_agent(**vars(opt))
    if opt.Loadmodel: agent.load(algo_name, BriefEnvName[opt.EnvIdex], opt.ModelIdex)

    if opt.render:
        while True:
            score = evaluate_policy(env, agent, 1)
            print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed) # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False

            '''Interact & train'''
            while not done:
                # epsilon-greedy exploration
                if total_steps < opt.random_steps:
                    a = env.action_space.sample()  # 返回的是 np.int64 而不是 python 原生 int
                else:
                    a = agent.select_action(s, deterministic=False)
                s_next, r, dw, tr, info = env.step(a) # dw: dead & win; tr: truncated
                done = (dw or tr)

                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next

                '''Update'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:  # 每交互 N 步，集中训练 N 次
                    for j in range(opt.update_every): agent.train()

                '''Noise decay & Record & Log'''
                if total_steps % 1000 == 0: agent.exp_noise *= opt.noise_decay
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent, turns = 3)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('noise', agent.exp_noise, global_step=total_steps)
                    print('EnvName:',BriefEnvName[opt.EnvIdex],'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', int(score))
                total_steps += 1

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    agent.save(algo_name,BriefEnvName[opt.EnvIdex],int(total_steps/1000))
    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()








