import numpy as np
import torch as th
import torch.nn as nn
import itertools
import collections
import random


# =========================
# Manager 层经验回放池
# =========================
class ManagerReplayBuffer:
    """
    用于存储 Manager（高层调度器）的交互经验
    每条经验对应一次任务分配决策
    """
    def __init__(self, capacity):
        # 使用 deque 实现固定容量的 FIFO 缓冲区
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, agent_type, action, reward, next_state, done, avail_agents):
        """
        存储一条 transition
        """
        self.buffer.append(
            (state, agent_type, action, reward, next_state, done, avail_agents)
        )

    def sample(self, batch_size):
        """
        随机采样一个 batch
        """
        transitions = random.sample(self.buffer, batch_size)
        state, agent_type, action, reward, next_state, done, avail_agents = zip(*transitions)

        return (
            np.array(state),
            np.array(agent_type),
            action,
            reward,
            np.array(next_state),
            done,
            avail_agents
        )

    def size(self):
        """
        返回当前 buffer 大小
        """
        return len(self.buffer)


# =========================
# 通用嵌入网络（Embedding）
# =========================
class Embedding(nn.Module):
    """
    用于将 agent / task 的原始特征映射到统一的 hidden embedding 空间
    """
    def __init__(self, input_dim, hidden_dim):
        super(Embedding, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = th.relu
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        x: (..., input_dim)
        return: (..., hidden_dim)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # 保证输出为二维张量 (N, hidden_dim)
        return x.reshape(-1, self.hidden_dim)


# =========================
# Manager 网络（Actor + Critic）
# =========================
class ManagerNet(nn.Module):
    """
    高层调度网络：
    - Actor：计算任务分配到各 agent 的概率
    - Critic：评估状态-动作的价值
    """
    def __init__(self, agent_dim, task_dim, hidden_dim, agent_type, device):
        super(ManagerNet, self).__init__()

        # 当前可用 agent 的属性（动态变化）
        self.agent_type = th.tensor(
            agent_type, dtype=th.float, device=device
        ).squeeze(0)

        # agent / task 的 embedding 网络
        self.agent_embed = Embedding(agent_dim, hidden_dim)
        self.task_embed = Embedding(task_dim, hidden_dim)

        # Critic 网络：分别对 agent 和 task 编码
        self.fc_task = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fc_agent = nn.Sequential(
            nn.Linear(agent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.hidden_dim = hidden_dim
        self.device = device

    def reset_agent(self, agent_type):
        """
        当 agent 数量或属性变化时，重置 agent embedding 输入
        """
        self.agent_type = th.tensor(
            agent_type, dtype=th.float, device=self.device
        ).squeeze(0)

    # =========================
    # Actor：计算分配概率
    # =========================
    def actor(self, task, avail_action):
        """
        task: (batch_size, task_dim)
        avail_action: (batch_size, num_agents) 可选 agent mask
        """

        avail_action = th.tensor(avail_action, dtype=th.float, device=self.device)
        if len(avail_action.shape) == 1:
            avail_action = avail_action.unsqueeze(0)

        task = th.tensor(task, dtype=th.float, device=self.device)

        # agent embedding: (num_agents, hidden_dim)
        agent_embedding = self.agent_embed(self.agent_type)

        # task embedding: (batch_size, hidden_dim)
        task_embedding = self.task_embed(task)

        # 点积注意力得分
        score = agent_embedding @ task_embedding.T / self.hidden_dim

        # 对不可用 agent 施加 mask
        probs = th.softmax(
            th.where(avail_action.T == 1, score, -1e6),
            dim=0
        )

        # 返回 (batch_size, num_agents)
        return probs.T

    # =========================
    # Critic：状态价值评估
    # =========================
    def critic(self, task):
        """
        计算每个 agent 对当前 task 的 Q 值
        """
        Q_value = self.fc_agent(self.agent_type) @ self.fc_task(task).reshape(-1, 1)
        return Q_value.T


# =========================
# Manager（SAC 训练逻辑）
# =========================
class Manager:
    """
    Manager 使用 SAC 框架进行训练
    """
    def __init__(
        self, state_dim, agent_dim, hidden_dim,
        actor_lr, critic_lr, alpha, tau, gamma,
        device, agent_type, model_dir
    ):
        # 主网络 & 目标网络
        self.net = ManagerNet(agent_dim, state_dim, hidden_dim, agent_type, device).to(device)
        self.target_net = ManagerNet(agent_dim, state_dim, hidden_dim, agent_type, device).to(device)
        self.target_net.load_state_dict(self.net.state_dict())

        self.loss = nn.MSELoss()

        # Actor / Critic 参数划分
        actor_params = itertools.chain(
            self.net.agent_embed.parameters(),
            self.net.task_embed.parameters()
        )
        critic_params = itertools.chain(
            self.net.fc_task.parameters(),
            self.net.fc_agent.parameters()
        )

        self.actor_optimizer = th.optim.Adam(actor_params, lr=actor_lr)
        self.critic_optimizer = th.optim.Adam(critic_params, lr=critic_lr)

        self.alpha = alpha          # 熵系数
        self.tau = tau              # 软更新系数
        self.gamma = gamma          # 折扣因子
        self.device = device

        self.critic_loss_list = []
        self.actor_loss_list = []
        self.model_dir = model_dir

    def reset_agent(self, agent_type):
        """
        同步更新主网络和目标网络的 agent 输入
        """
        self.net.reset_agent(agent_type)
        self.target_net.reset_agent(agent_type)

    def select(self, state, avail_action, evaluate=False):
        """
        根据策略选择 action
        """
        probs = self.net.actor(state, avail_action)

        if evaluate:
            action = probs.argmax()
        else:
            action_dist = th.distributions.Categorical(probs)
            action = action_dist.sample()

        return action.cpu().detach().item()

    def soft_update(self, net, target_net):
        """
        SAC 的软更新
        """
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save(self):
        th.save(self.net.state_dict(), self.model_dir + "/model.pt")

    def load(self):
        model_path = self.model_dir + "/model.pt"
        self.net.load_state_dict(th.load(model_path))
        self.target_net.load_state_dict(th.load(model_path))

    # =========================
    # SAC 更新
    # =========================
    def update(self, transition_dict):
        """
        使用一批 transition 更新 Actor 和 Critic
        """

        states = th.tensor(transition_dict['states'], dtype=th.float).flatten().to(self.device)
        agent_type = th.tensor(transition_dict['agent_type'], dtype=th.float).to(self.device)
        actions = th.tensor(transition_dict['actions'], dtype=th.long).to(self.device)
        rewards = th.tensor(transition_dict['rewards'], dtype=th.float).to(self.device)
        next_states = th.tensor(transition_dict['next_states'], dtype=th.float).flatten().to(self.device)
        dones = th.tensor(transition_dict['dones'], dtype=th.float).to(self.device)
        avail_agents = th.tensor(transition_dict['avail_agents'], dtype=th.float).flatten().to(self.device)

        # 更新 agent 输入
        self.reset_agent(agent_type)

        # -------- Critic 更新 --------
        next_probs = self.net.actor(next_states, avail_agents)
        next_log_probs = th.log(next_probs + 1e-8)
        entropy = -th.sum(next_probs * next_log_probs, dim=1, keepdim=True)

        q_value = self.net.critic(next_states)
        v_value = th.sum(next_probs * q_value, dim=1, keepdim=True).flatten()
        next_value = v_value + 0.01 * entropy

        td_target = rewards + self.gamma * next_value * (1 - dones)

        critic_q_values = self.net.critic(states).reshape(1, -1).gather(1, actions.reshape(1, 1))
        critic_loss = self.loss(td_target, critic_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------- Actor 更新 --------
        avail_agents.scatter_(0, actions, 1)
        probs = self.net.actor(states, avail_agents)
        log_probs = th.log(probs + 1e-8)
        entropy = -th.sum(probs * log_probs, dim=1, keepdim=True)

        q_value = self.target_net.critic(states)
        v_value = th.sum(probs * q_value, dim=1, keepdim=True)

        actor_loss = th.mean(-0.01 * entropy - v_value)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        self.soft_update(self.net, self.target_net)
