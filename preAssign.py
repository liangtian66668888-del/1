import numpy as np
import torch as th
import torch.nn as nn
import collections
import random


# =========================
# 预分配阶段（PreAssign / DGPA）经验回放池
# =========================
class PreAssignReplayBuffer:
    def __init__(self, capacity):
        # 使用 deque 作为固定容量 FIFO 缓冲区
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, agent_type, avail_agent, avail_task):
        """
        存储一条 transition
        state: 当前状态（通常是任务集合状态）
        action: 预分配动作（通常是 agent->task 的 one-hot/矩阵）
        reward: 奖励（可为即时奖励或外部定义的回报）
        agent_type: agent 属性/特征集合
        avail_agent: 可用 agent mask
        avail_task: 可用 task mask
        """
        self.buffer.append((state, action, reward, agent_type, avail_agent, avail_task))

    def sample(self, batch_size):
        """
        随机采样 batch_size 条 transition
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, agent_type, avail_agent, avail_task = zip(*transitions)
        return np.array(state), action, reward, agent_type, avail_agent, avail_task

    def size(self):
        """
        当前 buffer 长度
        """
        return len(self.buffer)


# =========================
# Embedding：将原始特征映射到隐藏空间
# =========================
class Embedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Embedding, self).__init__()
        # 两层 MLP 作为 embedding
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = th.relu
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (..., input_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # 返回: (..., hidden_dim)
        return x


# =========================
# Actor：普通 MLP（用于消融版本）
# =========================
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = th.relu
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # 输出 logits / score（具体后续如何使用由调用方决定）
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# =========================
# Critic：普通 MLP（用于消融版本）
# =========================
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = th.relu
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # 输出 Q 值（或 value），维度由 output_dim 决定
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# =========================
# PreAssignActor：预分配策略网络（注意力式匹配）
# =========================
class PreAssignActor(nn.Module):
    def __init__(self, task_dim, agent_dim, hidden_dim, device):
        super(PreAssignActor, self).__init__()
        # agent 与 task 分别做 embedding
        self.agent_embed = Embedding(agent_dim, hidden_dim)
        self.task_embed = Embedding(task_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, tasks, agent_type, avail_task):
        """
        tasks: (batch_size, task_num, task_dim) 或可 reshape 的等价结构
        agent_type: (batch_size, agent_num, agent_dim) 或可 reshape 的等价结构
        avail_task: (batch_size, task_num) 任务可用 mask（1 表示可用）
        """
        # 将 avail_task 放到 device
        avail_task = th.tensor(avail_task, device=self.device)

        # agent_embedding: (batch_size, agent_num, hidden_dim)
        # 注：这里 reshape(-1, agent_num, hidden_dim) 依赖输入最后两维含义
        agent_embedding = self.agent_embed(agent_type).reshape(
            -1, agent_type.shape[-2], self.hidden_dim
        )  # batch_size,agent_num,agent_dim -> batch_size,agent_num,hidden_dim

        # task_embedding: (batch_size, task_num, hidden_dim)
        task_embedding = self.task_embed(tasks).reshape(
            -1, tasks.shape[-2], self.hidden_dim
        )  # batch_size,task_num,hidden_dim

        # 计算点积匹配分数：
        # bmm: (batch, agent_num, hidden_dim) @ (batch, hidden_dim, task_num) -> (batch, agent_num, task_num)
        # permute -> (batch, task_num, agent_num) 方便按 task 维度做 mask/softmax
        # / hidden_dim 做 scaling（类似注意力机制的缩放）
        # th.where：对不可用任务位置用极小值屏蔽
        logits = th.softmax(
            th.where(
                avail_task.unsqueeze(-1).repeat(1, 1, agent_type.shape[-2]) == 1,
                th.bmm(agent_embedding, task_embedding.permute(0, 2, 1)).permute(0, 2, 1)
                / th.tensor(self.hidden_dim, dtype=th.long, device=self.device).flatten(),
                -999999.
            ),
            dim=1  # 在 task_num 维度 softmax：形成对任务的分布
        )

        # 返回 (batch_size, agent_num, task_num)
        return logits.permute(0, 2, 1)


# =========================
# PreAssignCritic：预分配 Q 值网络（矩阵形式）
# =========================
class PreAssignCritic(nn.Module):
    def __init__(self, task_dim, agent_dim, hidden_dim, device):
        super(PreAssignCritic, self).__init__()
        self.agent_embed = Embedding(agent_dim, hidden_dim)
        self.task_embed = Embedding(task_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, tasks, agent_type):
        # 输出 Q 矩阵：
        # (batch, agent_num, hidden_dim) @ (batch, hidden_dim, task_num) -> (batch, agent_num, task_num)
        return th.bmm(self.agent_embed(agent_type), self.task_embed(tasks).permute(0, 2, 1))


# =========================
# MIX：将每个 agent 的 value 混合成全局 value（注意力式 mixing）
# =========================
class MIX(nn.Module):
    def __init__(self, agent_dim, hidden_dim):
        super(MIX, self).__init__()
        # 这里 q/k/v 是对 agent_type 的编码，用于构造注意力 mixing 权重
        self.q = nn.Sequential(nn.Linear(agent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.k = nn.Sequential(nn.Linear(agent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.v1 = nn.Sequential(nn.Linear(agent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.v2 = nn.Sequential(nn.Linear(agent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()

    def forward(self, agent_type, value, avail_agents):
        """
        agent_type: (batch_size, agent_num, agent_dim)
        value: (batch_size, agent_num)  每个 agent 的局部价值
        avail_agents: (batch_size, agent_num) 可用 agent mask
        """
        # mask 不可用 agent 的属性，避免参与 mixing
        agent_type = agent_type * avail_agents.unsqueeze(-1)  # batch_size,agent_num,agent_dim

        key = self.k(agent_type)    # batch_size,agent_num,hidden_dim
        query = self.q(agent_type)  # batch_size,agent_num,hidden_dim

        # 注意力权重： (batch, agent_num, agent_num)
        score = th.softmax(
            th.bmm(key, query.permute(0, 2, 1))
            / th.sqrt(th.tensor(self.hidden_dim, dtype=th.int, device=agent_type.device)),
            dim=-1
        )

        value_w = self.v1(agent_type)  # batch_size,agent_num,hidden_dim
        w = th.bmm(score, value_w)     # batch_size,agent_num,hidden_dim

        # 用局部 value 对 w 做加权聚合： (batch,1,agent_num)@(batch,agent_num,hidden_dim) -> (batch,1,hidden_dim)
        x = th.bmm(value.unsqueeze(1), w)  # x:batch_size,1,hidden_dim
        x = self.relu(x)
        x = self.fc(x)  # x:batch_size,1,1

        # 输出 (batch,1)
        return x.reshape(-1, 1)


# =========================
# PreAssign：预分配模块（Actor + Critic + MIX）训练封装
# =========================
class PreAssign:
    def __init__(self, task_dim, agent_dim, hidden_dim, actor_lr, critic_lr, alpha, tau, gamma, device, model_dir):
        # Actor：输出 agent->task 概率分布
        self.actor = PreAssignActor(task_dim, agent_dim, hidden_dim, device).to(device)
        # Critic：输出 Q(agent, task) 矩阵
        self.critic = PreAssignCritic(task_dim, agent_dim, hidden_dim, device).to(device)
        # MIX：聚合局部价值为全局价值
        self.mix = MIX(agent_dim, hidden_dim).to(device)

        # 目标 critic 网络（用于稳定训练）
        self.target_critic = PreAssignCritic(task_dim, agent_dim, hidden_dim, device).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.hidden_dim = hidden_dim
        self.task_dim = task_dim

        self.loss = nn.MSELoss()

        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.mix_optimizer = th.optim.Adam(self.mix.parameters(), lr=critic_lr)

        # 超参数（alpha/gamma/tau 等）
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma

        self.critic_loss_list = []
        self.actor_loss_list = []
        self.model_dir = model_dir
        self.device = device

    def select(self, tasks, agent_type, avail_agent, avail_task, evaluate=False):
        """
        根据 actor 输出的概率，为每个 agent 选一个 task
        返回 one-hot action 矩阵：(agent_num, task_num)
        """
        avail_agent = th.tensor(avail_agent)
        tasks = th.tensor(tasks, dtype=th.float, device=self.device)
        agent_type = th.tensor(agent_type, dtype=th.float, device=self.device)

        # probs: (agent_num, task_num)（这里 squeeze(0) 假设 batch=1）
        probs = self.actor(tasks, agent_type, avail_task).squeeze(0)
        na, nt = probs.shape

        # evaluate=True 时用 argmax；否则按分布采样
        if evaluate:
            action = probs.argmax(axis=-1)
        else:
            action_dist = th.distributions.Categorical(probs)
            action = action_dist.sample()

        # 转 one-hot： (agent_num, task_num)
        action = th.eye(nt)[action.cpu()]  # na,nt

        # 不可用 agent 的动作置 0
        action = action * avail_agent.unsqueeze(-1)  # na,nt

        return action.cpu().detach()

    def update(self, transition_dict):
        """
        训练更新：
        - critic：拟合 td_target（这里 td_target = rewards）
        - actor：最大化 total_v_value + entropy（以 loss 形式写成 -entropy - value）
        """
        rewards = th.tensor(transition_dict['rewards'], dtype=th.float).flatten().to(self.device)
        batch_size = len(rewards)

        # states reshape 为 (batch, task_num, task_dim)
        states = th.tensor(transition_dict['states'], dtype=th.float).reshape(batch_size, -1, self.task_dim).to(self.device)

        # actions: (batch, agent_num, task_num) one-hot
        actions = th.tensor(transition_dict['actions'], dtype=th.long).to(self.device)

        # agent_types: (batch, agent_num, agent_dim)
        agent_types = th.tensor(transition_dict['agent_types'], dtype=th.float).to(self.device)

        # 可用 agent/task mask
        avail_agents = th.tensor(transition_dict['avail_agents'], dtype=th.float).reshape(batch_size, -1).to(self.device)
        avail_tasks = th.tensor(transition_dict['avail_tasks'], dtype=th.float).reshape(batch_size, -1).to(self.device)

        # TD target：当前实现是直接用 rewards（无 bootstrap）
        td_target = rewards

        # Critic：Q(agent,task) * action(one-hot) -> 选中 task 的 Q -> sum(-1) 得到每个 agent 的 Q
        critic_q_values = (self.critic(states, agent_types) * actions).sum(-1)  # batch,agent_num

        # MIX：聚合为全局 Q_total
        critic_total_q_values = self.mix(agent_types, critic_q_values, avail_agents).reshape(td_target.shape)  # batch,1

        # Critic loss
        critic_loss = self.loss(td_target, critic_total_q_values)
        self.critic_loss_list.append(critic_loss.cpu().detach().item())

        self.critic_optimizer.zero_grad()
        self.mix_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.mix_optimizer.step()

        # Actor：输出概率分布
        probs = self.actor(states, agent_types, avail_tasks)
        log_probs = th.log(probs + 1e-8)

        # entropy：这里按你原始写法对 probs/log_probs 做聚合，鼓励探索
        entropy = -th.sum(probs.sum(-1) * log_probs.sum(-1), dim=1, keepdim=True)

        # 用 target_critic 估计 Q 值（并结合 action）
        q_values = (self.target_critic(states, agent_types) * actions)

        # v_value：对 task 维做期望
        v_value = th.sum(probs * q_values, dim=-1)

        # MIX 聚合为全局 V_total
        total_v_value = self.mix(agent_types, v_value, avail_agents)

        # Actor loss：最大化（value + entropy），以最小化形式写出
        actor_loss = th.mean(-0.01 * entropy - total_v_value)
        self.actor_loss_list.append(actor_loss.cpu().detach().item())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新 target critic
        self.soft_update(self.critic, self.target_critic)

    def soft_update(self, net, target_net):
        """
        目标网络软更新：target = (1-tau)*target + tau*online
        """
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self):
        """
        保存 actor/critic/mix
        """
        actor_path = self.model_dir + "/actor.pt"
        critic_path = self.model_dir + "/critic.pt"
        mix_path = self.model_dir + "/mix.pt"
        th.save(self.actor.state_dict(), actor_path)
        th.save(self.critic.state_dict(), critic_path)
        th.save(self.mix.state_dict(), mix_path)

    def load(self):
        """
        加载 actor/critic/mix
        """
        actor_path = self.model_dir + "/actor.pt"
        critic_path = self.model_dir + "/critic.pt"
        mix_path = self.model_dir + "/mix.pt"
        self.actor.load_state_dict(th.load(actor_path))
        self.critic.load_state_dict(th.load(critic_path))
        self.mix.load_state_dict(th.load(mix_path))


# =========================
# PreAssignNormalCritic：消融版本（整体 critic）
# =========================
class PreAssignNormalCritic:
    def __init__(self, task_dim, agent_dim, hidden_dim, actor_lr, critic_lr, alpha, tau, gamma, device, model_dir, agent_num, task_num):
        # Actor 仍用注意力式预分配
        self.actor = PreAssignActor(task_dim, agent_dim, hidden_dim, device).to(device)

        # Critic：整体输入（tasks + agents + actions）输出一个全局标量
        self.critic = Critic(task_dim * task_num + agent_dim * agent_num + agent_num * task_num, hidden_dim, 1).to(device)
        self.target_critic = Critic(task_dim * task_num + agent_dim * agent_num + agent_num * task_num, hidden_dim, 1).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.hidden_dim = hidden_dim
        self.task_dim = task_dim
        self.loss = nn.MSELoss()

        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma

        self.critic_loss_list = []
        self.actor_loss_list = []
        self.model_dir = model_dir
        self.device = device

        # 工程 trick：给概率加偏置（避免出现 0 概率导致 log 问题）
        self.bias = 0.5

    def select(self, tasks, agent_type, avail_agent, avail_task, evaluate=False):
        # 与 PreAssign.select 类似：返回 one-hot action
        avail_agent = th.tensor(avail_agent)
        tasks = th.tensor(tasks, dtype=th.float, device=self.device)
        agent_type = th.tensor(agent_type, dtype=th.float, device=self.device)
        probs = self.actor(tasks, agent_type, avail_task).squeeze(0)
        na, nt = probs.shape
        if evaluate:
            action = probs.argmax(axis=-1)
        else:
            action_dist = th.distributions.Categorical(probs)
            action = action_dist.sample()
        action = th.eye(nt)[action.cpu()]   # na,nt
        action = action * avail_agent.unsqueeze(-1)  # na,nt
        return action.cpu().detach()

    def update(self, transition_dict):
        rewards = th.tensor(transition_dict['rewards'], dtype=th.float).flatten().to(self.device)
        batch_size = len(rewards)

        states = th.tensor(transition_dict['states'], dtype=th.float).reshape(batch_size, -1, self.task_dim).to(self.device)
        actions = th.tensor(transition_dict['actions'], dtype=th.long).to(self.device)
        agent_types = th.tensor(transition_dict['agent_types'], dtype=th.float).to(self.device)
        avail_tasks = th.tensor(transition_dict['avail_tasks'], dtype=th.float).reshape(batch_size, -1).to(self.device)

        td_target = rewards

        # 整体 critic 输入拼接，输出全局 Q
        critic_q_values = self.critic(
            th.cat((states.reshape(batch_size, -1), agent_types.reshape(batch_size, -1), actions.reshape(batch_size, -1)), dim=-1)
        )
        critic_loss = self.loss(td_target, critic_q_values)
        self.critic_loss_list.append(critic_loss.cpu().detach().item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor 更新
        probs = self.actor(states, agent_types, avail_tasks) + self.bias
        probs[probs > 1] = 1

        prob_action = (probs * actions).sum(-1)
        non_zero_rows = [row[row != 0] for row in prob_action]
        non_zero_rows = [row if row.numel() > 0 else th.tensor([1.]) for row in non_zero_rows]
        row_products = [th.prod(row).to(self.device) for row in non_zero_rows]
        prob = th.stack(row_products).reshape(-1, 1)

        log_probs = th.log(prob + 1e-8)
        entropy = -th.sum(prob * log_probs, dim=1, keepdim=True)

        q_values = (self.target_critic(
            th.cat((states.reshape(batch_size, -1), agent_types.reshape(batch_size, -1), actions.reshape(batch_size, -1)), dim=-1)
        ))
        v_value = th.sum(prob * q_values, dim=-1)

        actor_loss = th.mean(-0.01 * entropy - v_value)
        self.actor_loss_list.append(actor_loss.cpu().detach().item())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.critic, self.target_critic)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self):
        # 保存 actor / critic
        actor_path = self.model_dir + "/actor_normal_critic.pt"
        critic_path = self.model_dir + "/critic_normal_critic.pt"
        mix_path = self.model_dir + "/mix_normal_critic.pt"
        th.save(self.actor.state_dict(), actor_path)
        th.save(self.critic.state_dict(), critic_path)
        th.save(self.mix.state_dict(), mix_path)

    def load(self):
        actor_path = self.model_dir + "/actor_normal_critic.pt"
        critic_path = self.model_dir + "/critic_normal_critic.pt"
        mix_path = self.model_dir + "/mix_normal_critic.pt"
        self.actor.load_state_dict(th.load(actor_path))
        self.critic.load_state_dict(th.load(critic_path))
        self.mix.load_state_dict(th.load(mix_path))


# =========================
# PreAssignNormal：消融版本（普通 actor/critic + mix）
# =========================
class PreAssignNormal:
    def __init__(self, task_dim, agent_dim, hidden_dim, actor_lr, critic_lr, alpha, tau, gamma, device, model_dir, agent_num, task_num):
        # 注意：这里 Actor 的构造参数
        self.actor = Actor(task_dim * task_num + agent_dim * agent_num, agent_num * task_num, hidden_dim, device).to(device)
        self.critic = Critic(task_dim * task_num + agent_dim * agent_num, hidden_dim, task_num * agent_num).to(device)
        self.mix = MIX(agent_dim, hidden_dim).to(device)
        self.target_critic = Critic(task_dim * task_num + agent_dim * agent_num, hidden_dim, task_num * agent_num).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.hidden_dim = hidden_dim
        self.task_dim = task_dim
        self.loss = nn.MSELoss()

        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.mix_optimizer = th.optim.Adam(self.mix.parameters(), lr=critic_lr)

        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma

        self.critic_loss_list = []
        self.actor_loss_list = []
        self.model_dir = model_dir
        self.device = device

    def select(self, tasks, agent_type, avail_agent, avail_task, evaluate=False):
        # 与前面相同：采样/贪心生成 one-hot action
        avail_agent = th.tensor(avail_agent)
        tasks = th.tensor(tasks, dtype=th.float, device=self.device)
        agent_type = th.tensor(agent_type, dtype=th.float, device=self.device)

        probs = self.actor(tasks, agent_type, avail_task).reshape(agent_type.shape[0], -1)
        na, nt = probs.shape

        if evaluate:
            action = probs.argmax(axis=-1)
        else:
            action_dist = th.distributions.Categorical(probs)
            action = action_dist.sample()

        action = th.eye(nt)[action.cpu()]   # na,nt
        action = action * avail_agent.unsqueeze(-1)  # na,nt
        return action.cpu().detach()

    def update(self, transition_dict):
        rewards = th.tensor(transition_dict['rewards'], dtype=th.float).flatten().to(self.device)
        batch_size = len(rewards)

        states = th.tensor(transition_dict['states'], dtype=th.float).reshape(batch_size, -1, self.task_dim).to(self.device)
        actions = th.tensor(transition_dict['actions'], dtype=th.long).to(self.device)
        agent_types = th.tensor(transition_dict['agent_types'], dtype=th.float).to(self.device)

        avail_agents = th.tensor(transition_dict['avail_agents'], dtype=th.float).reshape(batch_size, -1).to(self.device)
        avail_tasks = th.tensor(transition_dict['avail_tasks'], dtype=th.float).reshape(batch_size, -1).to(self.device)

        td_target = rewards

        # critic 输出 reshape 成 action 同形状，再与 one-hot action 相乘，得到选中 task 的 Q
        critic_q_values = (
            self.critic(th.cat((states.reshape(batch_size, -1), agent_types.reshape(batch_size, -1)), dim=-1))
            .reshape(actions.shape) * actions
        ).sum(-1)

        critic_total_q_values = self.mix(agent_types, critic_q_values, avail_agents).reshape(td_target.shape)
        critic_loss = self.loss(td_target, critic_total_q_values)
        self.critic_loss_list.append(critic_loss.cpu().detach().item())

        self.critic_optimizer.zero_grad()
        self.mix_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.mix_optimizer.step()

        probs = self.actor(states, agent_types, avail_tasks)
        log_probs = th.log(probs + 1e-8)
        entropy = -th.sum(probs.sum(-1) * log_probs.sum(-1), dim=1, keepdim=True)

        q_values = (
            self.target_critic(th.cat((states.reshape(batch_size, -1), agent_types.reshape(batch_size, -1)), dim=-1))
            .reshape(actions.shape) * actions
        )
        v_value = th.sum(probs * q_values, dim=-1)
        total_v_value = self.mix(agent_types, v_value, avail_agents)

        actor_loss = th.mean(-0.01 * entropy - total_v_value)
        self.actor_loss_list.append(actor_loss.cpu().detach().item())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.critic, self.target_critic)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self):
        actor_path = self.model_dir + "/actor_normal_critic.pt"
        critic_path = self.model_dir + "/critic_normal_critic.pt"
        mix_path = self.model_dir + "/mix_normal_critic.pt"
        th.save(self.actor.state_dict(), actor_path)
        th.save(self.critic.state_dict(), critic_path)
        th.save(self.mix.state_dict(), mix_path)

    def load(self):
        actor_path = self.model_dir + "/actor_normal_critic.pt"
        critic_path = self.model_dir + "/critic_normal_critic.pt"
        mix_path = self.model_dir + "/mix_normal_critic.pt"
        self.actor.load_state_dict(th.load(actor_path))
        self.critic.load_state_dict(th.load(critic_path))
        self.mix.load_state_dict(th.load(mix_path))
