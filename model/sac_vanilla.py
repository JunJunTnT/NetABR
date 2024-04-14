import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
max_action = 1
min_action = -1
# RENDER=False
EP_MAX = 500
EP_LEN = 500
GAMMA = 0.95
q_lr = 5e-3
value_lr = 5e-3
policy_lr = 5e-4
BATCH = 256
tau = 1e-2

def softmax(x):
    """ softmax function """
    x = np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
    return x

def adjust_bw_co(co: list):
    for co_index in range(len(co)):
        if co[co_index] < 0.05:
            max_index = np.argmax(co)
            co[max_index] -= 0.1
            co[co_index] += 0.1
    return co


class ActorNet(nn.Module):
    def __init__(self, inp, outp):
        super(ActorNet, self).__init__()
        self.in_to_y1 = nn.Linear(inp, 256)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        self.y1_to_y2 = nn.Linear(256, 256)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(256, outp)
        self.out.weight.data.normal_(0, 0.1)
        self.std_out = nn.Linear(256, outp)
        self.std_out.weight.data.normal_(0, 0.1)

    def forward(self, inputstate):
        inputstate = self.in_to_y1(inputstate)
        inputstate = F.relu(inputstate)
        inputstate = self.y1_to_y2(inputstate)
        inputstate = F.relu(inputstate)
        mean = max_action * torch.tanh(self.out(inputstate))  # 输出概率分布的均值mean
        log_std = self.std_out(inputstate)  # softplus激活函数的值域>0
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mean, std


class CriticNet(nn.Module):
    def __init__(self, input, output):
        super(CriticNet, self).__init__()
        # q1
        self.in_to_y1 = nn.Linear(input + output, 256)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        self.y1_to_y2 = nn.Linear(256, 256)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(256, 1)
        self.out.weight.data.normal_(0, 0.1)
        # q2
        self.q2_in_to_y1 = nn.Linear(input + output, 256)
        self.q2_in_to_y1.weight.data.normal_(0, 0.1)
        self.q2_y1_to_y2 = nn.Linear(256, 256)
        self.q2_y1_to_y2.weight.data.normal_(0, 0.1)
        self.q2_out = nn.Linear(256, 1)
        self.q2_out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        inputstate = torch.cat((s, a), dim=1)
        # q1
        q1 = self.in_to_y1(inputstate)
        q1 = F.relu(q1)
        q1 = self.y1_to_y2(q1)
        q1 = F.relu(q1)
        q1 = self.out(q1)
        # q2
        q2 = self.in_to_y1(inputstate)
        q2 = F.relu(q2)
        q2 = self.y1_to_y2(q2)
        q2 = F.relu(q2)
        q2 = self.out(q2)
        return q1, q2


class Memory():
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.mem = np.zeros((capacity, dims))
        self.memory_counter = 0

    '''存储记忆'''

    def store_transition(self, s, a, r, s_):
        tran = np.hstack((s, a, r, s_))  # 把s,a,r,s_困在一起，水平拼接
        index = self.memory_counter % self.capacity  # 除余得索引
        self.mem[index, :] = tran  # 给索引存值，第index行所有列都为其中一次的s,a,r,s_；mem会是一个capacity行，（s+a+r+s_）列的数组
        self.memory_counter += 1

    '''随机从记忆库里抽取'''

    def sample(self, n):
        assert self.memory_counter >= self.capacity, '记忆库没有存满记忆'
        sample_index = np.random.choice(self.capacity, n)  # 从capacity个记忆里随机抽取n个为一批，可得到抽样后的索引号
        new_mem = self.mem[sample_index, :]  # 由抽样得到的索引号在所有的capacity个记忆中  得到记忆s，a，r，s_
        return new_mem


class Actor():
    def __init__(self, state_dim: int, action_dim: int):
        self.action_net = ActorNet(state_dim, action_dim)  # 这只是均值mean
        self.optimizer = torch.optim.Adam(self.action_net.parameters(), lr=policy_lr)

    def choose_action(self, s):
        inputstate = torch.FloatTensor(s)
        mean, std = self.action_net(inputstate)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, min_action, max_action)
        action = action.detach().numpy()
        # action = np.array(softmax(action))
        # action = adjust_bw_co(action)
        return action

    def evaluate(self, s):
        inputstate = torch.FloatTensor(s)
        mean, std = self.action_net(inputstate)
        dist = torch.distributions.Normal(mean, std)
        noise = torch.distributions.Normal(0, 1)
        z = noise.sample()
        action = torch.tanh(mean + std * z)
        action = torch.clamp(action, min_action, max_action)
        action_logprob = dist.log_prob(mean + std * z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, action_logprob, z, mean, std

    def learn(self, actor_loss):
        loss = actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Entroy():
    def __init__(self, action_dim: int):
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=q_lr)

    def learn(self, entroy_loss):
        loss = entroy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Critic():
    def __init__(self, state_dim: int, action_dim: int):
        self.critic_v, self.target_critic_v = CriticNet(state_dim, action_dim), CriticNet(state_dim,
                                                                                          action_dim)  # 改网络输入状态，生成一个Q值
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=value_lr, eps=1e-5)
        self.lossfunc = nn.MSELoss()

    def soft_update(self):
        for target_param, param in zip(self.target_critic_v.parameters(), self.critic_v.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_v(self, s, a):
        return self.critic_v(s, a)

    def learn(self, current_q1, current_q2, target_q):
        loss = self.lossfunc(current_q1, target_q) + self.lossfunc(current_q2, target_q)
        # print("critic loss:" + str(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class SAC:
    def __init__(self, memory_capacity: int, state_dim: int, action_dim: int, model_dir=None):
        self.actor = Actor(state_dim=state_dim, action_dim=action_dim)
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim)
        self.entroy = Entroy(action_dim=action_dim)
        self.memory_capacity = memory_capacity
        self.M = Memory(memory_capacity, 2 * state_dim + action_dim + 1)
        self.all_ep_r = []
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_dir = model_dir

    def save_model(self):
        torch.save(obj=self.actor.action_net.state_dict(), f=self.model_dir + 'actor.pth')
        torch.save(obj=self.critic.critic_v.state_dict(), f=self.model_dir + 'critic_v.pth')
        torch.save(obj=self.critic.target_critic_v.state_dict(), f=self.model_dir + 'target_critic_v.pth')

    def load_model(self, model_dir: str):
        self.actor.action_net.load_state_dict(torch.load(model_dir + 'actor.pth'))
        self.critic.critic_v.load_state_dict(torch.load(model_dir + 'critic_v.pth'))
        self.critic.target_critic_v.load_state_dict(torch.load(model_dir + 'target_critic_v.pth'))

    def update(self, epoch: int):
        if self.M.memory_counter <= self.memory_capacity:
            return
        if epoch % 50 != 0:
            return
        b_M = self.M.sample(BATCH)
        b_s = b_M[:, :self.state_dim]
        b_a = b_M[:, self.state_dim: self.state_dim + self.action_dim]
        b_r = b_M[:, -self.state_dim - 1: -self.state_dim]
        b_s_ = b_M[:, -self.state_dim:]
        b_s = torch.FloatTensor(b_s)
        b_a = torch.FloatTensor(b_a)
        b_r = torch.FloatTensor(b_r)
        b_s_ = torch.FloatTensor(b_s_)
        new_action, log_prob_, z, mean, log_std = self.actor.evaluate(b_s_)
        target_q1, target_q2 = self.critic.get_v(b_s_, new_action)
        target_q = b_r + GAMMA * (torch.min(target_q1, target_q2) - self.entroy.alpha * log_prob_)
        current_q1, current_q2 = self.critic.get_v(b_s, b_a)

        self.critic.learn(current_q1, current_q2, target_q.detach())
        t = target_q.detach()
        a, log_prob, _, _, _ = self.actor.evaluate(b_s)
        q1, q2 = self.critic.get_v(b_s, a)
        q = torch.min(q1, q2)
        actor_loss = (self.entroy.alpha * log_prob - q).mean()
        self.actor.learn(actor_loss)
        alpha_loss = -(self.entroy.log_alpha.exp() * (log_prob + self.entroy.target_entropy).detach()).mean()
        self.entroy.learn(alpha_loss)
        self.entroy.alpha = self.entroy.log_alpha.exp()
        # 软更新
        self.critic.soft_update()
        # print('actor loss:' + str(actor_loss))
