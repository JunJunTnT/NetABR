import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from model.graph_layers import GraphConvLayer

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
    def __init__(self, state_dim, action_dim, n_agents, hidden_size=256, pool_type='avg'):
        super(CriticNet, self).__init__()
        # q1

        self.q1_gc1 = GraphConvLayer(state_dim + action_dim, hidden_size)
        self.q1_nn_gc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.q1_gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.q1_nn_gc2 = nn.Linear(hidden_size, hidden_size)

        self.q1_V = nn.Linear(hidden_size, 1)
        self.q1_V.weight.data.normal_(0, 0.1)

        # q2
        self.q2_gc1 = GraphConvLayer(state_dim + action_dim, hidden_size)
        self.q2_nn_gc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.q2_gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.q2_nn_gc2 = nn.Linear(hidden_size, hidden_size)

        self.q2_V = nn.Linear(hidden_size, 1)
        self.q2_V.weight.data.normal_(0, 0.1)
        self.n_agents = n_agents

        self.adj = (torch.ones(self.n_agents, self.n_agents) - torch.eye(self.n_agents)) / self.n_agents

        self.pool_type = pool_type

    def forward(self, s, a):
        sa = torch.cat((s, a), dim=2)
        # group_size = 4
        # group_id = [1, 1, 2, 2, 3, 3, 4, 4]
        # group_id = torch.Tensor([(np.array(group_id) / group_size).reshape(self.n_agents, -1)] * sa.shape[0])
        # sa = torch.cat((group_id, sa), dim=-1)

        q1 = F.relu(self.q1_gc1(sa, self.adj))
        q1 = F.relu(self.q1_nn_gc1(sa)) + q1
        q1 = q1 / (1. * self.n_agents)
        q1 = F.relu(self.q1_gc2(q1, self.adj))
        q1 = F.relu(self.q1_nn_gc2(q1)) + q1
        q1 = q1 / (1. * self.n_agents)
        if self.pool_type == 'avg':
            ret1 = q1.mean(1)
        elif self.pool_type == 'max':
            ret1, _ = q1.max(1)
        q1 = self.q1_V(ret1)

        # q2
        q2 = F.relu(self.q2_gc1(sa, self.adj))
        q2 = F.relu(self.q2_nn_gc1(sa)) + q2
        q2 = q2 / (1. * self.n_agents)
        q2 = F.relu(self.q2_gc2(q2, self.adj))
        q2 = F.relu(self.q2_nn_gc2(q2)) + q2
        q2 = q2 / (1. * self.n_agents)
        if self.pool_type == 'avg':
            ret2 = q2.mean(1)
        elif self.pool_type == 'max':
            ret2, _ = q2.max(1)
        q2 = self.q2_V(ret2)

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
        return action.detach().numpy()

    def evaluate(self, s):
        state_vec = torch.FloatTensor(s)
        mean, std = self.action_net(state_vec)
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
    def __init__(self, state_dim: int, action_dim: int, n_agents: int):
        self.critic_v, self.target_critic_v = CriticNet(state_dim, action_dim, n_agents), CriticNet(state_dim,
                                                                                                    action_dim,
                                                                                                    n_agents)
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=value_lr, eps=1e-5)
        self.lossfunc = nn.MSELoss()

    def soft_update(self):
        for target_param, param in zip(self.target_critic_v.parameters(), self.critic_v.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_v(self, s, a):
        return self.critic_v(s, a)

    def learn(self, current_q1, current_q2, target_q):
        loss = self.lossfunc(current_q1, target_q) + self.lossfunc(current_q2, target_q)
        print("critic loss:" + str(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class SAC:
    def __init__(self, memory_capacity: int, state_dim: int, action_dim: int, model_dir=None):
        self.action_dim = action_dim
        self.num_agents = action_dim
        self.actor = Actor(state_dim=state_dim, action_dim=action_dim)
        self.critic = Critic(state_dim=int(state_dim / self.num_agents), action_dim=int(action_dim / self.num_agents),
                             n_agents=self.num_agents)
        self.entroy = Entroy(action_dim=action_dim)
        self.memory_capacity = memory_capacity
        self.M = Memory(memory_capacity, 2 * state_dim + action_dim + 1)
        self.all_ep_r = []
        self.state_dim = state_dim
        self.model_dir = model_dir

    def save_model(self):
        torch.save(obj=self.actor.action_net.state_dict(), f=self.model_dir + 'actor.pth')
        torch.save(obj=self.critic.critic_v.state_dict(), f=self.model_dir + 'critic_v.pth')
        torch.save(obj=self.critic.target_critic_v.state_dict(), f=self.model_dir + 'target_critic_v.pth')

    def load_model(self, model_dir):
        self.actor.action_net.load_state_dict(torch.load(model_dir + 'actor.pth'))
        self.critic.critic_v.load_state_dict(torch.load(model_dir + 'critic_v.pth'))
        self.critic.target_critic_v.load_state_dict(torch.load(model_dir + 'target_critic_v.pth'))

    def update(self, epoch: int):
        if self.M.memory_counter <= self.memory_capacity:
            return
        if epoch % 50 != 0:
            return
        with torch.autograd.set_detect_anomaly(True):
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

            target_q1, target_q2 = self.critic.get_v(
                b_s_.view(-1, self.num_agents, int(self.state_dim / self.num_agents)),
                new_action.view(-1, self.num_agents, 1))
            target_q = b_r + GAMMA * (torch.min(target_q1, target_q2) - self.entroy.alpha * log_prob_)

            current_q1, current_q2 = self.critic.get_v(
                b_s.view(-1, self.num_agents, int(self.state_dim / self.num_agents)), b_a.view(-1, self.num_agents, 1))
            self.critic.learn(current_q1, current_q2, target_q.detach())

            a, log_prob, _, _, _ = self.actor.evaluate(b_s)

            q1, q2 = self.critic.get_v(b_s.view(-1, self.num_agents, int(self.state_dim / self.num_agents)),
                                       a.view(-1, self.num_agents, 1))
            q = torch.min(q1, q2)
            actor_loss = (self.entroy.alpha * log_prob - q).mean()

            self.actor.learn(actor_loss)
            alpha_loss = -(self.entroy.log_alpha.exp() * (log_prob + self.entroy.target_entropy).detach()).mean()
            self.entroy.learn(alpha_loss)
            self.entroy.alpha = self.entroy.log_alpha.exp()
            # 软更新
            self.critic.soft_update()
            print(actor_loss)
