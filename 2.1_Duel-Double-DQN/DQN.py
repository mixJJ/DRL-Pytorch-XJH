import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import copy


def build_net(layer_shape, activation, output_activation):  # 注意这个函数只适用于1D state, 因为没有flatten的拉平操作
	'''Build networks with For loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)

class Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):  # hid_shape = (200, 200)
		super(Q_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]
		self.Q = build_net(layers, nn.ReLU, nn.Identity)  # 要保存模型的结构作为属性, 会自动调用 forward 应用这些属性

	def forward(self, s):
		q = self.Q(s)
		return q


class Duel_Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Duel_Q_Net, self).__init__()
		layers = [state_dim] + list(hid_shape)
		self.hidden = build_net(layers, nn.ReLU, nn.ReLU)
		self.V = nn.Linear(hid_shape[-1], 1)  # -1 代表取最后一个元素
		self.A = nn.Linear(hid_shape[-1], action_dim)

	def forward(self, s):
		s = self.hidden(s)
		Adv = self.A(s)
		V = self.V(s)
		Q = V + (Adv - torch.mean(Adv, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
		return Q


class ReplayBuffer(object):  # 适用于当 state 是 1D
	def __init__(self, state_dim, dvc, max_size=int(1e6)):
		self.max_size = max_size  # 定义成属性 这样在下面的 def add 和 def sample 也可以调用 而不仅在 __init__() 内部可见
		self.dvc = dvc
		self.ptr = 0  # 添加经验时的索引位置
		self.size = 0  # 实际经验个数

		self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)  # 直接 =dvc 也行
		self.a = torch.zeros((max_size, 1), dtype=torch.long, device=self.dvc)  # torch.long 就是 torch.int64
		self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)  # 默认 float32
		self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
		self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)  # dw: dead&win; tr: truncated

	def add(self, s, a, r, s_next, dw):
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		# s 必须是 np.ndarray, np.int64不行, torch.from_numpy zero-copy, 只需移到GPU时copy一次, 不支持device=device; torch.tensor CPU上复制一次 移到GPU又复制一次
		self.a[self.ptr] = a  # 将 np.int64/int 直接赋值给 int64 tensor, PyTorch 允许把标量赋值给 GPU tensor 的单个元素, 隐式创建一个临时 tensor 然后 to(device) 拷贝到 GPU, 然后销毁临时 tensor
		# self.a[self.ptr] = torch.tensor(a, dtype=torch.long, device=self.dvc)  # 其实和 to(device) 等价的
		self.r[self.ptr] = r  # 将原生标量 float 直接赋值给 float32 tensor, PyTorch 允许把标量赋值给 GPU tensor 的单个元素, 并自动搬到 GPU 上
		# self.r[self.ptr] = torch.tensor(r, dtype=torch.float, device=self.dvc)
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw  # 将原生标量 bool 直接赋值给 bool tensor, PyTorch 允许把标量赋值给 GPU tensor 的单个元素, 并自动搬到 GPU 上
		# self.dw[self.ptr] = torch.tensor(dw, dtype=torch.bool, device=self.dvc)

		self.ptr = (self.ptr + 1) % self.max_size  # % 是 取模运算符 (modulo operator), 意思是 "求余数"
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		# ind = torch.randperm(self.size, device=self.dvc)[:batch_size]  # 从 [0, 1, ..., size - 1] 中随机打乱后取前 batch_size 个索引
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))  # 无放回抽取的话会拖慢速度 这里允许抽到重复的经验
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]



class DQN_agent(object):
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005  # Soft Update(软更新) target_net, 0.005 新参数 1-0.005 旧参数
		self.replay_buffer = ReplayBuffer(self.state_dim, self.dvc, max_size=int(1e6))
		if self.Duel:
			self.q_net = Duel_Q_Net(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
		else:
			self.q_net = Q_Net(self.state_dim, self.action_dim, (self.net_width, self.net_width)).to(self.dvc)
		self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
		self.q_target = copy.deepcopy(self.q_net)
		# Freeze target networks with respect to optimizers (only update via Polyak averaging)
		for p in self.q_target.parameters(): p.requires_grad = False


	def select_action(self, state, deterministic):  # only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
			if deterministic:
				a = self.q_net(state).argmax().item()
			else:
				if np.random.rand() < self.exp_noise:
					a = np.random.randint(0,self.action_dim)
				else:
					a = self.q_net(state).argmax().item()
		return a


	def train(self):
		s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

		'''Compute the target Q value'''
		with torch.no_grad():
			if self.Double:
				argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1) # 确定动作, .argmax(dim=1) 在列取最大值的index, 会消去列这个维度 .unsqueeze(-1) 在最后一个增加一个维度 变为 [batch_size,1]
				max_q_next = self.q_target(s_next).gather(1,argmax_a)  # gather 在 dim=1 按 argmax_a 这个 index 来找 (action)
			else:
				max_q_next = self.q_target(s_next).max(1)[0].unsqueeze(1) # max(1) 消去了列这个维度, 为了之后的计算 .unsqueeze(1) 在 dim=1 增加一个维度 变为 [batch_size,1]
			target_Q = r + (~dw) * self.gamma * max_q_next # 这就是 buffer 加入 dw 的目的, 用于这里的计算 dw: die or win, bool 可以参与数学运算 自动转换成0 1, ~表示取反

		# Get current Q estimates
		current_q = self.q_net(s)
		current_q_a = current_q.gather(1,a)

		q_loss = F.mse_loss(current_q_a, target_Q)
		self.q_net_optimizer.zero_grad()
		q_loss.backward()
		self.q_net_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self,algo,EnvName,steps):
		torch.save(self.q_net.state_dict(), "./model/{}_{}_{}.pth".format(algo,EnvName,steps))

	def load(self,algo,EnvName,steps):
		self.q_net.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps),map_location=self.dvc))
		self.q_target.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps),map_location=self.dvc))







