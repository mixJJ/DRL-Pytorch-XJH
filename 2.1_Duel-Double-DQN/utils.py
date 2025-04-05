def evaluate_policy(env, agent, turns = 3):  # turns=3 是默认值不是强制值,传其他参数会覆盖默认值
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)  # 并不是 evaluate_policy 能访问 class DQN_agent, 而是此处传入的 DQN_agent instance(实例) 本身"携带"了 select_action 方法
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores/turns)


#You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool): # Is x an instance of type T?
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'): # lower() 函数把所有大写字母转成小写
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        print('Wrong Input.')
        raise