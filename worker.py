import numpy as np
import torch as th
import torch.nn as nn
import collections
import random


# =========================
# Worker 层经验回放池
# =========================
class WorkerReplayBuffer:
    def __init__(self, capacity):
        # 使用 deque 保存固定容量的 transition
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward):
        """
        存储一条 transition
        state: worker 的输入状态（通常是 task 特征或局部状态）
        action: worker 输出动作（这里是一个连续值）
        reward: 该动作对应的即时奖励
        """
        self.buffer.append((state, action, reward))

    def sample(self, batch_size):
        """
        从 buffer 中随机采样 batch_size 条数据
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward = zip(*transitions)
        return np.array(state), action, reward

    def size(self):
        """
        返回当前 buffer 大小
        """
        return len(self.buffer)


# =========================
# Worker Actor 网络：输入 state/task 输出连续动作（标量）
# =========================
class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ActorNet, self).__init__()
        # 简单的 MLP：state_dim -> hidden -> ... -> 1
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 注意：这里定义了 fc2，但 forward 里未使用
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = th.relu  # 注意：这里定义了 relu，但 forward 里未使用

    def forward(self, x):
        """
        x: (..., state_dim)
        输出: (..., 1) 且通过 sigmoid 限制在 (0,1) 范围
        """
        x = self.fc1(x)

        # 这里使用 sigmoid 而不是 relu，表示输出在 (0,1) 之间（用于表示资源需求比例等）
        x = th.sigmoid(x)

        # 输出层
        x = self.fc3(x)

        # 再做一次 sigmoid，将最终动作限制在 (0,1)
        x = th.sigmoid(x)
        return x


# =========================
# Worker Critic 网络：输入 (state, action) 输出 Q(s,a)
# =========================
class CriticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CriticNet, self).__init__()
        # 输入维度一般是 state_dim + action_dim（这里 action_dim=1）
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 同样定义了 fc2，但 forward 未使用（保持原样，不改）
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = th.relu

    def forward(self, x, a):
        """
        x: state 张量
        a: action 张量
        输出: Q(s,a)
        """
        # 如果输入是一维向量，则扩展 batch 维度
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        # 拼接 state 和 action，形成 critic 输入
        cat = th.cat([x, a], dim=-1)

        x = self.fc1(cat)
        x = self.relu(x)
        x = self.fc3(x)

        # 输出做 relu（保持原样），意味着 Q 值被限制为非负
        x = self.relu(x)
        return x


# =========================
# Worker：执行层（连续动作）策略 + 训练逻辑
# =========================
class Worker:
    def __init__(self, state_dim, hidden_dim, sigma, actor_lr, critic_lr, tau, gamma, device, model_dir, buffer_size,
                 sigma_decay):
        """
        state_dim: worker 输入维度（任务特征维度/局部状态维度）
        hidden_dim: MLP 隐藏层维度
        sigma: 探索噪声初始标准差（高斯噪声）
        actor_lr / critic_lr: 学习率
        tau: target 网络软更新系数
        gamma: 折扣因子（当前 update 中未使用，保持原样）
        buffer_size: 回放池容量
        sigma_decay: 噪声衰减系数
        """
        # 在线网络
        self.actor = ActorNet(state_dim, hidden_dim).to(device)
        self.critic = CriticNet(state_dim + 1, hidden_dim).to(device)  # state_dim + action_dim(=1)

        # 目标网络（DDPG 风格）
        self.target_actor = ActorNet(state_dim, hidden_dim).to(device)
        self.target_critic = CriticNet(state_dim + 1, hidden_dim).to(device)

        # 初始化目标网络参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 优化器
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.sigma = sigma  # 探索噪声
        self.tau = tau  # 软更新系数
        self.device = device
        self.model_dir = model_dir

        self.loss = nn.MSELoss()
        self.actor_loss_list = []
        self.critic_loss_list = []

        # Worker 专用回放池
        self.buffer = WorkerReplayBuffer(buffer_size)

        # 噪声衰减系数
        self.sigma_decay = sigma_decay

    def demand(self, task):
        """
        根据当前 task 状态输出一个连续动作（需求/强度/资源量等）
        并加入高斯噪声用于探索
        """
        # 衰减 sigma，但不低于 0.05（保持一定探索）
        self.sigma = self.sigma * self.sigma_decay if self.sigma > 0.05 else 0.05

        # task 转 tensor，并 reshape 为 (1, state_dim)
        task = th.tensor(task, dtype=th.float, device=self.device).reshape(1, -1)

        # actor 输出动作（0~1）
        action = self.actor(task).item()

        # 加入高斯噪声进行探索
        action = action + self.sigma * np.random.randn()

        # 截断：动作不允许为负
        action = action if action > 0 else 0
        return action

    def soft_update(self, net, target_net):
        """
        目标网络软更新：
        target = (1-tau)*target + tau*online
        """
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        """
        一次更新：同时更新 critic 和 actor
        transition_dict 需包含：
        - states
        - actions
        - rewards
        """
        states = th.tensor(transition_dict['states'], dtype=th.float, device=self.device)
        actions = th.tensor(transition_dict['actions'], dtype=th.float, device=self.device).reshape(-1, 1)
        rewards = th.tensor(transition_dict['rewards'], dtype=th.float, device=self.device).reshape(-1, 1)

        # 这里 q_targets 直接等于 rewards（无 bootstrap / 无 next_state）
        q_targets = rewards

        # -------- critic 更新：拟合 Q(s,a) -> reward --------
        critic_loss = self.loss(self.critic(states, actions), q_targets)
        self.critic_loss_list.append(critic_loss.cpu().detach().item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------- actor 更新：最大化 Q(s, actor(s)) --------
        # 等价于最小化 -Q
        actor_loss = -th.mean(self.critic(states, self.actor(states)))
        self.actor_loss_list.append(actor_loss.cpu().detach().item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -------- 软更新目标网络 --------
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

    def save(self):
        """
        保存 worker actor / critic 参数
        """
        actor_path = self.model_dir + "/worker_actor.pt"
        critic_path = self.model_dir + "/worker_critic.pt"
        th.save(self.actor.state_dict(), actor_path)
        th.save(self.critic.state_dict(), critic_path)

    def load(self):
        """
        加载 worker actor / critic 参数
        """
        actor_path = self.model_dir + "/worker_actor.pt"
        critic_path = self.model_dir + "/worker_critic.pt"
        self.actor.load_state_dict(th.load(actor_path))
        self.critic.load_state_dict(th.load(critic_path))
