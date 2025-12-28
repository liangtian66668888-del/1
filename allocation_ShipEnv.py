import numpy as np
from manager import Manager, ManagerReplayBuffer
from worker import Worker
from preAssign import PreAssign, PreAssignReplayBuffer, PreAssignNormalCritic
from env.ShipEnv import ShipEnv as BaseShipEnv
import env.config as config


class Allocation:
    def __init__(self, args):
        self.epoch = 0
        self.manager_buffer_size = args.manager_buffer_size
        self.manager_min_size = args.manager_min_size
        self.manager_batch_size = args.manager_batch_size
        self.preassign_buffer_size = args.preassign_buffer_size
        self.preassign_min_size = args.preassign_min_size
        self.preassign_batch_size = args.preassign_batch_size
        self.hidden_dim = args.hidden_dim
        self.max_epoch = args.max_epoch
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.tau = args.tau
        self.sigma = args.worker_sigma
        self.sigma_decay = args.sigma_decay_rate
        self.device = args.device
        self.model_dir = args.model_dir
        self.agent_num = args.agent_num
        self.agent_dim = args.agent_dim
        self.task_num = args.task_num
        self.task_dim = args.task_dim
        self.avail_agent = [np.zeros(self.agent_num, dtype=np.int8) for _ in range(self.task_num)]
        self.agent_type = args.agent_type
        self.preassign_buffer = PreAssignReplayBuffer(self.preassign_buffer_size)
        self.preassign = PreAssign(self.task_dim, self.agent_dim, self.hidden_dim, self.actor_lr, self.critic_lr,
                                   self.alpha, self.tau, self.gamma, self.device, self.model_dir)
        self.manager_buffer = ManagerReplayBuffer(self.manager_buffer_size)
        self.manager = Manager(self.task_dim, self.agent_dim, self.hidden_dim, self.actor_lr, self.critic_lr,
                               self.alpha, self.tau, self.gamma, self.device, self.agent_type, self.model_dir)

    def chose_agents(self, task_id, task, evaluate, render=True):
        avail_agents = self.avail_agent[task_id]
        state = task.copy()
        done = 0
        returns = 0
        action_list = []
        while not done:
            action = self.manager.select(state, avail_action=avail_agents, evaluate=evaluate)
            next_state, reward, done, avail_agents = self.manager_step(action, task_id)
            transition_dict = {'states': state.copy(), 'agent_type': self.agent_type.copy(), 'actions': action,
                               'next_states': next_state.copy(), 'rewards': reward, 'dones': done,
                               'avail_agents': avail_agents}
            self.manager.update(transition_dict)
            state = next_state
            returns += reward
            action_list.append(action)
        if returns <= 0:
            action_list = []
            returns = 0
        if render:
            if len(self.manager.actor_loss_list) > 0:
                print("manager{} actor loss: {}".format(task_id, self.manager.actor_loss_list[-1]),
                      " manager{} critic loss: {}".format(task_id, self.manager.critic_loss_list[-1]))
                print("manager{} accepts this task!".format(task_id), " allocation", action_list, " return:", returns)
        return action_list, returns

    def select(self, state_, avail_task_, evaluate=False, render=False):
        state = state_.copy()
        self.task = state[:self.task_num * self.task_dim].reshape(self.task_num, self.task_dim)
        self.task_copy = self.task.copy()
        avail_task = (self.task.sum(axis=1) != 0) * avail_task_
        allocation = [[] for _ in range(self.task_num)]
        if sum(avail_task) == 0:
            return allocation
        self.agent_type = state[
                          self.task_num * self.task_dim:self.task_num * self.task_dim + self.agent_num * self.agent_dim].reshape(
            self.agent_num, self.agent_dim)
        avail_agents = state[self.task_num * self.task_dim + self.agent_num * self.agent_dim:]
        self.avail_agent = np.array(
            self.preassign.select(self.task.copy(), self.agent_type, avail_agents, avail_task, evaluate).T)
        pre_assign = self.avail_agent.copy()
        total_returns = 0
        for task_id, task in enumerate(self.task):
            if avail_task[task_id] == 1:
                action_list, returns = self.chose_agents(task_id, task, evaluate, render)
                allocation[task_id].extend(action_list)
                total_returns += returns

        self.preassign_buffer.add(self.task_copy.copy(), pre_assign.T, total_returns, self.agent_type.copy(),
                                  avail_agents, avail_task)
        if self.preassign_buffer.size() > self.preassign_min_size:
            pre_states, pre_actions, pre_rewards, pre_agent_types, pre_avail_agents, pre_avail_tasks = self.preassign_buffer.sample(
                self.preassign_batch_size)
            transition_dict = {'states': pre_states, 'actions': pre_actions, 'rewards': pre_rewards,
                               "agent_types": pre_agent_types, "avail_agents": pre_avail_agents,
                               "avail_tasks": pre_avail_tasks}
            self.preassign.update(transition_dict)
        return allocation

    def manager_step(self, action, task_id):
        true_action = self.agent_type[action]
        self.task[task_id][1:] = self.task[task_id][1:] - true_action[1:]
        self.task[task_id][1:] = np.where(self.task[task_id][1:] <= 0, 0, self.task[task_id][1:])
        self.avail_agent[task_id][action] = 0
        done = 0
        if np.all(self.task[task_id][1:] == 0) or (sum(self.avail_agent[task_id]) == 0):
            done = 1
        reward = self.task[task_id][0] - true_action[0] if np.all(self.task[task_id][1:] == 0) else -true_action[0]
        return self.task[task_id].copy(), reward, done, self.avail_agent[task_id].copy()

    def save(self):
        self.manager.save()
        self.preassign.save()

    def load(self):
        self.manager.load()
        self.preassign.load()


class SelfishAllocation:
    def __init__(self, args):
        self.epoch = 0
        self.manager_buffer_size = args.manager_buffer_size
        self.manager_min_size = args.manager_min_size
        self.manager_batch_size = args.manager_batch_size
        self.preassign_buffer_size = args.preassign_buffer_size
        self.preassign_min_size = args.preassign_min_size
        self.preassign_batch_size = args.preassign_batch_size
        self.hidden_dim = args.hidden_dim
        self.max_epoch = args.max_epoch
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.tau = args.tau
        self.sigma = args.worker_sigma
        self.sigma_decay = args.sigma_decay_rate
        self.device = args.device
        self.model_dir = args.model_dir
        self.agent_num = args.agent_num
        self.agent_dim = args.agent_dim
        self.task_num = args.task_num
        self.task_dim = args.task_dim
        self.worker_list = []
        self.worker_buffer_size = args.worker_buffer_size
        self.worker_min_size = args.worker_min_size
        self.worker_batch_size = args.worker_batch_size
        self.avail_agent = [np.zeros(self.agent_num, dtype=np.int8) for _ in range(self.task_num)]
        self.agent_type = args.agent_type
        self.preassign_buffer = PreAssignReplayBuffer(self.preassign_buffer_size)
        self.preassign = PreAssign(self.task_dim, self.agent_dim, self.hidden_dim, self.actor_lr, self.critic_lr,
                                   self.alpha, self.tau, self.gamma, self.device, self.model_dir)
        self.manager_buffer = ManagerReplayBuffer(self.manager_buffer_size)
        self.manager = Manager(self.task_dim, self.agent_dim, self.hidden_dim, self.actor_lr, self.critic_lr,
                               self.alpha, self.tau, self.gamma, self.device, self.agent_type, self.model_dir)
        for _ in self.agent_type:
            w = Worker(self.agent_dim, self.hidden_dim, self.sigma, self.actor_lr, self.critic_lr, self.tau, self.gamma,
                       self.device, self.model_dir, self.worker_buffer_size, self.sigma_decay)
            self.worker_list.append(w)

    def chose_agents(self, task_id, task, evaluate, render=True):
        avail_agents = self.avail_agent[task_id]
        init_avail_agents = avail_agents.copy()
        idxs = np.where(init_avail_agents == 1)[0]

        state = task.copy()
        demands = np.zeros(len(self.agent_type))
        for idx in idxs:
            demands[idx] = self.worker_list[idx].demand(task.copy())

        self.agent_type[idxs, 0] = demands[idxs].copy()
        self.manager.reset_agent(self.agent_type.copy())
        state = task.copy()
        done = 0
        returns = 0
        action_list = []
        agent_rewards = np.zeros(len(self.agent_type))
        while not done:
            action = self.manager.select(state, avail_action=avail_agents, evaluate=evaluate)
            next_state, reward, done, avail_agents = self.manager_step(action, task_id)
            transition_dict = {'states': state.copy(), 'agent_type': self.agent_type.copy(), 'actions': action,
                               'next_states': next_state.copy(), 'rewards': reward, 'dones': done,
                               'avail_agents': avail_agents}
            self.manager.update(transition_dict)
            state = next_state
            returns += reward
            action_list.append(action)
            agent_rewards[action] = self.agent_type[action, 0]
        if returns <= 0:
            action_list = []
            returns = 0
            agent_rewards = np.zeros(len(self.agent_type))
        for idx in idxs:
            transition_dict = {'states': self.task_copy[task_id].copy(), 'actions': demands[idx].copy(),
                               'rewards': agent_rewards[idx].copy()}
            self.worker_list[idx].update(transition_dict)
        if render:
            for idx in range(len(self.worker_list)):
                if len(self.worker_list[idx].actor_loss_list) > 0:
                    print("worker {} actor loss: {}".format(idx, self.worker_list[idx].actor_loss_list[-1]),
                          " worker critic loss: {}".format(self.worker_list[idx].critic_loss_list[-1]))
            if len(self.manager.actor_loss_list) > 0:
                print("manager{} actor loss: {}".format(task_id, self.manager.actor_loss_list[-1]),
                      " manager{} critic loss: {}".format(task_id, self.manager.critic_loss_list[-1]))
                print("manager{} accepts this task!".format(task_id), " allocation", action_list, " return:", returns)
        return action_list, returns

    def select(self, state_, avail_task_, evaluate=False, render=False):
        state = state_.copy()
        self.task = state[:self.task_num * self.task_dim].reshape(self.task_num, self.task_dim)
        self.task_copy = self.task.copy()
        avail_task = (self.task.sum(axis=1) != 0) * avail_task_
        allocation = [[] for _ in range(self.task_num)]
        if sum(avail_task) == 0:
            return allocation
        self.agent_type = state[
                          self.task_num * self.task_dim:self.task_num * self.task_dim + self.agent_num * self.agent_dim].reshape(
            self.agent_num, self.agent_dim)
        avail_agents = state[self.task_num * self.task_dim + self.agent_num * self.agent_dim:]
        self.avail_agent = np.array(
            self.preassign.select(self.task.copy(), self.agent_type, avail_agents, avail_task, evaluate).T)
        pre_assign = self.avail_agent.copy()
        total_returns = 0
        for task_id, task in enumerate(self.task):
            if avail_task[task_id] == 1:
                action_list, returns = self.chose_agents(task_id, task, evaluate, render)
                allocation[task_id].extend(action_list)
                total_returns += returns

        self.preassign_buffer.add(self.task_copy.copy(), pre_assign.T, total_returns, self.agent_type.copy(),
                                  avail_agents, avail_task)
        if self.preassign_buffer.size() > self.preassign_min_size:
            pre_states, pre_actions, pre_rewards, pre_agent_types, pre_avail_agents, pre_avail_tasks = self.preassign_buffer.sample(
                self.preassign_batch_size)
            transition_dict = {'states': pre_states, 'actions': pre_actions, 'rewards': pre_rewards,
                               "agent_types": pre_agent_types, "avail_agents": pre_avail_agents,
                               "avail_tasks": pre_avail_tasks}
            self.preassign.update(transition_dict)
        return allocation

    def manager_step(self, action, task_id):
        true_action = self.agent_type[action]
        self.task[task_id][1:] = self.task[task_id][1:] - true_action[1:]
        self.task[task_id][1:] = np.where(self.task[task_id][1:] <= 0, 0, self.task[task_id][1:])
        self.avail_agent[task_id][action] = 0
        done = 0
        if np.all(self.task[task_id][1:] == 0) or (sum(self.avail_agent[task_id]) == 0):
            done = 1
        reward = self.task[task_id][0] - true_action[0] if np.all(self.task[task_id][1:] == 0) else -true_action[0]
        return self.task[task_id].copy(), reward, done, self.avail_agent[task_id].copy()

    def save(self):
        self.manager.save()
        self.preassign.save()

    def load(self):
        self.manager.load()
        self.preassign.load()


class AllocationWoPre:
    def __init__(self, args):
        self.epoch = 0
        self.manager_buffer_size = args.manager_buffer_size
        self.manager_min_size = args.manager_min_size
        self.manager_batch_size = args.manager_batch_size
        self.worker_buffer_size = args.worker_buffer_size
        self.worker_min_size = args.worker_min_size
        self.worker_batch_size = args.worker_batch_size
        self.hidden_dim = args.hidden_dim
        self.max_epoch = args.max_epoch
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.tau = args.tau
        self.sigma = args.worker_sigma
        self.sigma_decay = args.sigma_decay_rate
        self.device = args.device
        self.evaluate = args.evaluate
        self.model_dir = args.model_dir
        self.agent_num = args.agent_num
        self.agent_dim = args.agent_dim
        self.task_num = args.task_num
        self.task_dim = args.task_dim
        self.avail_agent = np.zeros(self.agent_num, dtype=np.int8)
        self.worker_list = []
        self.agent_type = args.agent_type
        self.random = args.random
        self.manager_buffer = ManagerReplayBuffer(self.manager_buffer_size)
        self.manager = Manager(self.task_dim, self.agent_dim, self.hidden_dim, self.actor_lr, self.critic_lr,
                               self.alpha, self.tau, self.gamma, self.device, self.agent_type, self.model_dir)

    def chose_agents(self, task_id, task):
        avail_agents = self.avail_agent
        state = task.copy()
        done = 0
        returns = 0
        action_list = []
        while not done:
            action = self.manager.select(state, avail_action=avail_agents)
            next_state, reward, done, avail_agents = self.manager_step(action, task_id)
            transition_dict = {'states': state.copy(), 'agent_type': self.agent_type.copy(), 'actions': action,
                               'next_states': next_state.copy(), 'rewards': reward, 'dones': done,
                               'avail_agents': avail_agents}
            self.manager.update(transition_dict)
            state = next_state
            returns += reward
            action_list.append(action)
        if returns <= 0:
            action_list = []
            returns = 0
        else:
            self.avail_agent = avail_agents
        return action_list

    def select(self, state, avail_task):
        self.task = state[:self.task_num * self.task_dim].reshape(self.task_num, self.task_dim)
        self.task_copy = self.task.copy()
        self.agent_type = state[
                          self.task_num * self.task_dim:self.task_num * self.task_dim + self.agent_num * self.agent_dim].reshape(
            self.agent_num, self.agent_dim)
        avail_agents = state[self.task_num * self.task_dim + self.agent_num * self.agent_dim:]
        avail_idx = np.where(avail_agents == 1)[0]
        self.avail_agent = np.zeros(self.agent_num, dtype=np.int8)
        self.avail_agent[avail_idx] = 1
        allocation = [[] for _ in range(self.task_num)]
        if self.random:
            shuffled_indices = np.random.permutation(self.task.shape[0])
            shuffle_task = self.task[shuffled_indices]
            for t_id, task in enumerate(shuffle_task):
                task_id = shuffled_indices[t_id]
                if avail_task[task_id] == 1:
                    allocation[task_id].extend(self.chose_agents(task_id, task))
        else:
            for task_id, task in enumerate(self.task):
                if avail_task[task_id] == 1:
                    allocation[task_id].extend(self.chose_agents(task_id, task))
        return allocation

    def manager_step(self, action, task_id):
        true_action = self.agent_type[action]
        self.task[task_id][1:] = self.task[task_id][1:] - true_action[1:]
        self.task[task_id] = np.where(self.task[task_id] <= 0, 0, self.task[task_id])
        self.avail_agent[action] = 0
        done = 0
        if np.all(self.task[task_id][1:] == 0) or (sum(self.avail_agent) == 0):
            done = 1
        reward = self.task[task_id][0] - true_action[0] if np.all(self.task[task_id][1:] == 0) else -true_action[0]
        return self.task[task_id].copy(), reward, done, self.avail_agent.copy()

    def save(self):
        self.manager.save()

    def load(self):
        self.manager.load()


class AllocationNormalCritic:
    def __init__(self, args):
        self.epoch = 0
        self.manager_buffer_size = args.manager_buffer_size
        self.manager_min_size = args.manager_min_size
        self.manager_batch_size = args.manager_batch_size
        self.preassign_buffer_size = args.preassign_buffer_size
        self.preassign_min_size = args.preassign_min_size
        self.preassign_batch_size = args.preassign_batch_size
        self.hidden_dim = args.hidden_dim
        self.max_epoch = args.max_epoch
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.tau = args.tau
        self.sigma = args.worker_sigma
        self.sigma_decay = args.sigma_decay_rate
        self.device = args.device
        self.model_dir = args.model_dir
        self.agent_num = args.agent_num
        self.agent_dim = args.agent_dim
        self.task_num = args.task_num
        self.task_dim = args.task_dim
        self.avail_agent = [np.zeros(self.agent_num, dtype=np.int8) for _ in range(self.task_num)]
        self.agent_type = args.agent_type
        self.preassign_buffer = PreAssignReplayBuffer(self.preassign_buffer_size)
        self.preassign = PreAssignNormalCritic(self.task_dim, self.agent_dim, self.hidden_dim, self.actor_lr,
                                               self.critic_lr, self.alpha, self.tau, self.gamma, self.device,
                                               self.model_dir, self.agent_num, self.task_num)
        self.manager_buffer = ManagerReplayBuffer(self.manager_buffer_size)
        self.manager = Manager(self.task_dim, self.agent_dim, self.hidden_dim, self.actor_lr, self.critic_lr,
                               self.alpha, self.tau, self.gamma, self.device, self.agent_type, self.model_dir)

    def chose_agents(self, task_id, task, evaluate, render=True):
        avail_agents = self.avail_agent[task_id]
        state = task.copy()
        done = 0
        returns = 0
        action_list = []
        while not done:
            action = self.manager.select(state, avail_action=avail_agents, evaluate=evaluate)
            next_state, reward, done, avail_agents = self.manager_step(action, task_id)
            transition_dict = {'states': state.copy(), 'agent_type': self.agent_type.copy(), 'actions': action,
                               'next_states': next_state.copy(), 'rewards': reward, 'dones': done,
                               'avail_agents': avail_agents}
            self.manager.update(transition_dict)
            state = next_state
            returns += reward
            action_list.append(action)
        if returns <= 0:
            action_list = []
            returns = 0
        if render:
            if len(self.manager.actor_loss_list) > 0:
                print("manager{} actor loss: {}".format(task_id, self.manager.actor_loss_list[-1]),
                      " manager{} critic loss: {}".format(task_id, self.manager.critic_loss_list[-1]))
                print("manager{} accepts this task!".format(task_id), " allocation", action_list, " return:", returns)
        return action_list, returns

    def select(self, state_, avail_task_, evaluate=False, render=False):
        state = state_.copy()
        self.task = state[:self.task_num * self.task_dim].reshape(self.task_num, self.task_dim)
        self.task_copy = self.task.copy()
        avail_task = (self.task.sum(axis=1) != 0) * avail_task_
        allocation = [[] for _ in range(self.task_num)]
        if sum(avail_task) == 0:
            return allocation
        self.agent_type = state[
                          self.task_num * self.task_dim:self.task_num * self.task_dim + self.agent_num * self.agent_dim].reshape(
            self.agent_num, self.agent_dim)
        avail_agents = state[self.task_num * self.task_dim + self.agent_num * self.agent_dim:]
        self.avail_agent = np.array(
            self.preassign.select(self.task.copy(), self.agent_type, avail_agents, avail_task, evaluate).T)
        pre_assign = self.avail_agent.copy()
        total_returns = 0
        for task_id, task in enumerate(self.task):
            if avail_task[task_id] == 1:
                action_list, returns = self.chose_agents(task_id, task, evaluate, render)
                allocation[task_id].extend(action_list)
                total_returns += returns

        self.preassign_buffer.add(self.task_copy.copy(), pre_assign.T, total_returns, self.agent_type.copy(),
                                  avail_agents, avail_task)
        if self.preassign_buffer.size() > self.preassign_min_size:
            pre_states, pre_actions, pre_rewards, pre_agent_types, pre_avail_agents, pre_avail_tasks = self.preassign_buffer.sample(
                self.preassign_batch_size)
            transition_dict = {'states': pre_states, 'actions': pre_actions, 'rewards': pre_rewards,
                               "agent_types": pre_agent_types, "avail_agents": pre_avail_agents,
                               "avail_tasks": pre_avail_tasks}
            self.preassign.update(transition_dict)
        return allocation

    def manager_step(self, action, task_id):
        true_action = self.agent_type[action]
        self.task[task_id][1:] = self.task[task_id][1:] - true_action[1:]
        self.task[task_id][1:] = np.where(self.task[task_id][1:] <= 0, 0, self.task[task_id][1:])
        self.avail_agent[task_id][action] = 0
        done = 0
        if np.all(self.task[task_id][1:] == 0) or (sum(self.avail_agent[task_id]) == 0):
            done = 1
        reward = self.task[task_id][0] - true_action[0] if np.all(self.task[task_id][1:] == 0) else -true_action[0]
        return self.task[task_id].copy(), reward, done, self.avail_agent[task_id].copy()

    def save(self):
        self.manager.save()
        self.preassign.save()

    def load(self):
        self.manager.load()
        self.preassign.load()


class AllocationNormal:
    def __init__(self, args):
        self.epoch = 0
        self.manager_buffer_size = args.manager_buffer_size
        self.manager_min_size = args.manager_min_size
        self.manager_batch_size = args.manager_batch_size
        self.preassign_buffer_size = args.preassign_buffer_size
        self.preassign_min_size = args.preassign_min_size
        self.preassign_batch_size = args.preassign_batch_size
        self.hidden_dim = args.hidden_dim
        self.max_epoch = args.max_epoch
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.tau = args.tau
        self.sigma = args.worker_sigma
        self.sigma_decay = args.sigma_decay_rate
        self.device = args.device
        self.model_dir = args.model_dir
        self.agent_num = args.agent_num
        self.agent_dim = args.agent_dim
        self.task_num = args.task_num
        self.task_dim = args.task_dim
        self.avail_agent = [np.zeros(self.agent_num, dtype=np.int8) for _ in range(self.task_num)]
        self.agent_type = args.agent_type
        self.preassign_buffer = PreAssignReplayBuffer(self.preassign_buffer_size)
        self.preassign = PreAssignNormalCritic(self.task_dim, self.agent_dim, self.hidden_dim, self.actor_lr,
                                               self.critic_lr, self.alpha, self.tau, self.gamma, self.device,
                                               self.model_dir, self.agent_num, self.task_num)
        self.manager_buffer = ManagerReplayBuffer(self.manager_buffer_size)
        self.manager = Manager(self.task_dim, self.agent_dim, self.hidden_dim, self.actor_lr, self.critic_lr,
                               self.alpha, self.tau, self.gamma, self.device, self.agent_type, self.model_dir)

    def chose_agents(self, task_id, task, evaluate, render=True):
        avail_agents = self.avail_agent[task_id]
        state = task.copy()
        done = 0
        returns = 0
        action_list = []
        while not done:
            action = self.manager.select(state, avail_action=avail_agents, evaluate=evaluate)
            next_state, reward, done, avail_agents = self.manager_step(action, task_id)
            transition_dict = {'states': state.copy(), 'agent_type': self.agent_type.copy(), 'actions': action,
                               'next_states': next_state.copy(), 'rewards': reward, 'dones': done,
                               'avail_agents': avail_agents}
            self.manager.update(transition_dict)
            state = next_state
            returns += reward
            action_list.append(action)
        if returns <= 0:
            action_list = []
            returns = 0
        if render:
            if len(self.manager.actor_loss_list) > 0:
                print("manager{} actor loss: {}".format(task_id, self.manager.actor_loss_list[-1]),
                      " manager{} critic loss: {}".format(task_id, self.manager.critic_loss_list[-1]))
                print("manager{} accepts this task!".format(task_id), " allocation", action_list, " return:", returns)
        return action_list, returns

    def select(self, state_, avail_task_, evaluate=False, render=False):
        state = state_.copy()
        self.task = state[:self.task_num * self.task_dim].reshape(self.task_num, self.task_dim)
        self.task_copy = self.task.copy()
        avail_task = (self.task.sum(axis=1) != 0) * avail_task_
        allocation = [[] for _ in range(self.task_num)]
        if sum(avail_task) == 0:
            return allocation
        self.agent_type = state[
                          self.task_num * self.task_dim:self.task_num * self.task_dim + self.agent_num * self.agent_dim].reshape(
            self.agent_num, self.agent_dim)
        avail_agents = state[self.task_num * self.task_dim + self.agent_num * self.agent_dim:]
        self.avail_agent = np.array(
            self.preassign.select(self.task.copy(), self.agent_type, avail_agents, avail_task, evaluate).T)
        pre_assign = self.avail_agent.copy()
        total_returns = 0
        for task_id, task in enumerate(self.task):
            if avail_task[task_id] == 1:
                action_list, returns = self.chose_agents(task_id, task, evaluate, render)
                allocation[task_id].extend(action_list)
                total_returns += returns

        self.preassign_buffer.add(self.task_copy.copy(), pre_assign.T, total_returns, self.agent_type.copy(),
                                  avail_agents, avail_task)
        if self.preassign_buffer.size() > self.preassign_min_size:
            pre_states, pre_actions, pre_rewards, pre_agent_types, pre_avail_agents, pre_avail_tasks = self.preassign_buffer.sample(
                self.preassign_batch_size)
            transition_dict = {'states': pre_states, 'actions': pre_actions, 'rewards': pre_rewards,
                               "agent_types": pre_agent_types, "avail_agents": pre_avail_agents,
                               "avail_tasks": pre_avail_tasks}
            self.preassign.update(transition_dict)
        return allocation

    def manager_step(self, action, task_id):
        true_action = self.agent_type[action]
        self.task[task_id][1:] = self.task[task_id][1:] - true_action[1:]
        self.task[task_id][1:] = np.where(self.task[task_id][1:] <= 0, 0, self.task[task_id][1:])
        self.avail_agent[task_id][action] = 0
        done = 0
        if np.all(self.task[task_id][1:] == 0) or (sum(self.avail_agent[task_id]) == 0):
            done = 1
        reward = self.task[task_id][0] - true_action[0] if np.all(self.task[task_id][1:] == 0) else -true_action[0]
        return self.task[task_id].copy(), reward, done, self.avail_agent[task_id].copy()

    def save(self):
        self.manager.save()
        self.preassign.save()

    def load(self):
        self.manager.load()
        self.preassign.load()


class ShipEnv:
    def __init__(self):
        self.env = BaseShipEnv()
        self.agent_num = len(config.agent_loc)
        self.agent_dim = 3  # [capability, x, y]
        self.task_num = len(config.works_loc)  # 最大任务数
        self.task_dim = 3  # [reward, x, y]
        self.epoch = 0
        self.max_steps = 200
        self.current_step = 0

        # === RBF reward shaping hyper-parameters ===
        # sigma: distance scale (grid units). larger => smoother, affects farther tasks
        self.rbf_sigma = 10.0
        # scale: amplifies dense RBF reward so it doesn't get drowned out by step penalty
        # scale applied after internal scaling
        self.rbf_scale = 1.0
        # internal scaling for RBF to keep episode returns in a stable range
        self.rbf_internal_scale = 0.01
        # per-step time penalty to encourage quicker completion
        self.step_penalty = 0.01
        # if True: use dense reward sum(phi); if False: use potential-difference shaping sum(phi-phi_prev)
        self.rbf_dense = True
        self._prev_phi = np.zeros(self.agent_num, dtype=np.float32)
        # 初始化agent_type: [capability, x, y]
        self.agent_type = np.zeros((self.agent_num, self.agent_dim))
        for i, pos in enumerate(config.agent_loc):
            self.agent_type[i] = [1.0, pos[0], pos[1]]  # capability=1.0, position

        # 初始化任务状态
        self.task_states = np.zeros((self.task_num, self.task_dim))
        # 存储任务的初始奖励值（用于计算total_reward）
        self.task_initial_rewards = np.zeros(self.task_num)
        # 跟踪上一时刻哪些任务在accidents中（用于检测任务完成）
        self.prev_task_in_accidents = np.zeros(self.task_num, dtype=bool)
        # 存储每个workNode的初始resourceNum（用于计算奖励）
        self.workNode_initial_resourceNum = {}

    def _calc_phi(self, agent_pos, task_pos, task_active):
        """Compute RBF potential for each agent w.r.t. active tasks.

        agent_pos: (N,2), task_pos: (M,2), task_active: (M,) bool
        returns: (N,) phi (sum of Gaussians), larger is better.
        """
        active_idx = np.where(task_active)[0]
        if active_idx.size == 0:
            return np.zeros(agent_pos.shape[0], dtype=np.float32)

        ap = agent_pos.astype(np.float32)
        tp = task_pos[active_idx].astype(np.float32)  # (K,2)
        diff = ap[:, None, :] - tp[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)  # (N,K)
        sigma2 = float(self.rbf_sigma) ** 2
        rbf = np.exp(-0.5 * d2 / sigma2)  # (N,K)
        return np.sum(rbf, axis=1).astype(np.float32)

    def reset(self):
        """重置环境并返回状态和info"""
        obs_n = self.env.reset()
        self.current_step = 0
        self.epoch = 0

        # 更新agent位置
        for i, agent in self.env.agents.items():
            self.agent_type[i][1] = agent.pos[0]
            self.agent_type[i][2] = agent.pos[1]

        # 初始化任务状态：事故点作为任务
        self.task_states = np.zeros((self.task_num, self.task_dim))
        self.task_initial_rewards = np.zeros(self.task_num)  # 重置初始奖励
        self.prev_task_in_accidents = np.zeros(self.task_num, dtype=bool)  # 重置任务状态跟踪

        # 保存每个workNode的初始resourceNum（在reset时，resourceNum已经重置为初始值）
        for i, work_node in enumerate(self.env.workNodes.values()):
            self.workNode_initial_resourceNum[i] = work_node.resourceNum

        for i, work_node in enumerate(self.env.workNodes.values()):
            # 检查任务是否在accidents中（使用更健壮的比较方式）
            work_pos = work_node.pos
            in_accidents = any(
                acc[0] == work_pos[0] and acc[1] == work_pos[1]
                for acc in self.env.accidents
            )
            if in_accidents:
                # 任务奖励基于初始资源需求，位置信息
                initial_resourceNum = self.workNode_initial_resourceNum[i]
                reward = initial_resourceNum * 10.0  # 奖励与初始资源需求相关
                self.task_states[i] = [reward, work_node.pos[0], work_node.pos[1]]
                self.task_initial_rewards[i] = reward  # 保存初始奖励
                self.prev_task_in_accidents[i] = True  # 标记任务在accidents中
            else:
                self.task_states[i] = [0, work_node.pos[0], work_node.pos[1]]
                self.task_initial_rewards[i] = 0
                self.prev_task_in_accidents[i] = False

        # 构建状态向量: [tasks, agent_types, avail_agents]
        state = np.concatenate([
            self.task_states.flatten(),
            self.agent_type.flatten(),
            np.ones(self.agent_num)  # 所有agent初始可用
        ])

        # 构建info字典
        task_pos = np.array([self.task_states[i][1:] for i in range(self.task_num)])
        agent_pos = np.array([self.agent_type[i][1:] for i in range(self.agent_num)])

        # === RBF: initialize previous potential ===
        task_active = (self.task_states[:, 0] > 0)
        self._prev_phi = self._calc_phi(agent_pos, task_pos, task_active)
        info = {
            "task_pos": task_pos,
            "agent_pos": agent_pos,
            "total_reward": 0.0
        }

        return state, info

    def step(self, action):
        """执行一步，action是agent的动作数组"""
        # 将action转换为ShipEnv期望的格式
        # take_action返回的action: 0=不动, 1=上, 2=下, 3=左, 4=右, 5=到达
        # ShipEnv期望: action_n[i][0]在[0,1]范围内，会被转换为0-3的离散动作
        # agent_speed映射: 0=[0,1](下), 1=[0,-1](上), 2=[-1,0](左), 3=[1,0](右)
        # ShipEnv转换: [0,0.125)或[0.875,1]->0(下), [0.125,0.375)->1(上), [0.375,0.625)->2(左), [0.625,0.875)->3(右)
        action_n = []
        for a in action:
            if a == 0:  # 不动
                action_n.append([0.0, 0.0])
            elif a == 1:  # 上 -> agent_speed[1] -> action_n[0]应该在[0.125, 0.375)
                action_n.append([0.25, 0.0])
            elif a == 2:  # 下 -> agent_speed[0] -> action_n[0]应该在[0, 0.125)或[0.875, 1]
                action_n.append([0.05, 0.0])
            elif a == 3:  # 左 -> agent_speed[2] -> action_n[0]应该在[0.375, 0.625)
                action_n.append([0.5, 0.0])
            elif a == 4:  # 右 -> agent_speed[3] -> action_n[0]应该在[0.625, 0.875)
                action_n.append([0.75, 0.0])
            elif a == 5:  # 到达/处理 -> 不动
                action_n.append([0.0, 0.0])
            else:
                action_n.append([0.0, 0.0])

        # 在执行环境步骤之前，保存当前任务状态（用于检测任务完成）
        # 注意：这里保存的是执行env.step()之前的状态
        prev_task_in_accidents_before_step = self.prev_task_in_accidents.copy()

        # 执行环境步骤
        obs_n, reward_n, done, Info = self.env.step(
            totalstep=self.max_steps,
            currstep=self.current_step,
            action_n=action_n
        )

        self.current_step += 1
        self.epoch = self.current_step

        # 更新agent位置
        for i, agent in self.env.agents.items():
            self.agent_type[i][1] = agent.pos[0]
            self.agent_type[i][2] = agent.pos[1]

        # 更新任务状态
        total_reward = 0.0
        for i, work_node in enumerate(self.env.workNodes.values()):
            # 检查任务是否在accidents中（使用更健壮的比较方式）
            work_pos = work_node.pos
            current_in_accidents = any(
                acc[0] == work_pos[0] and acc[1] == work_pos[1]
                for acc in self.env.accidents
            )

            if current_in_accidents:
                # 任务仍在进行
                # 如果是新任务（之前不在accidents中），保存初始奖励
                if not prev_task_in_accidents_before_step[i]:
                    # 使用初始resourceNum计算奖励，而不是当前可能已减少的resourceNum
                    initial_resourceNum = self.workNode_initial_resourceNum[i]
                    reward = initial_resourceNum * 10.0
                    self.task_initial_rewards[i] = reward
                    self.task_states[i][0] = reward
                else:
                    # 任务已存在，保持初始奖励值不变（不随resourceNum减少而减少）
                    self.task_states[i][0] = self.task_initial_rewards[i]
                self.task_states[i][1] = work_node.pos[0]
                self.task_states[i][2] = work_node.pos[1]
                self.prev_task_in_accidents[i] = True
            else:
                # 任务不在accidents中
                # 如果之前任务在accidents中，说明任务刚完成
                if prev_task_in_accidents_before_step[i] and self.task_initial_rewards[i] > 0:
                    # 任务刚完成，给予初始奖励（而不是当前可能已减少的奖励）
                    task_reward = self.task_initial_rewards[i]
                    total_reward += task_reward
                    self.task_initial_rewards[i] = 0  # 标记已奖励，允许同一位置出现新任务
                    # 调试信息（可选，用于验证任务完成检测）
                    # print(f"Task {i} completed! Reward: {task_reward}, Total reward so far: {total_reward}")
                self.task_states[i][0] = 0
                self.task_states[i][1] = work_node.pos[0]
                self.task_states[i][2] = work_node.pos[1]
                self.prev_task_in_accidents[i] = False

        # 计算奖励：基于agent到达事故点的奖励
        # === RBF reward: ignore base env reward_n, use RBF + completion bonus ===
        task_pos = np.array([self.task_states[i][1:] for i in range(self.task_num)], dtype=np.float32)
        agent_pos = np.array([self.agent_type[i][1:] for i in range(self.agent_num)], dtype=np.float32)
        task_active = (self.task_states[:, 0] > 0)

        phi = self._calc_phi(agent_pos, task_pos, task_active)

        if self.rbf_dense:
            r_rbf = float(np.sum(phi))
        else:
            r_rbf = float(np.sum(phi - self._prev_phi))
        self._prev_phi = phi

        # scale dense shaping reward (RBF) without shrinking sparse task completion reward
        rbf_scaled = self.rbf_scale * (self.rbf_internal_scale * r_rbf)
        reward = rbf_scaled + total_reward - self.step_penalty
        # 构建下一状态
        state = np.concatenate([
            self.task_states.flatten(),
            self.agent_type.flatten(),
            np.ones(self.agent_num)  # 假设所有agent都可用
        ])

        # 更新info
        task_pos = np.array([self.task_states[i][1:] for i in range(self.task_num)])
        agent_pos = np.array([self.agent_type[i][1:] for i in range(self.agent_num)])

        info = {
            "task_pos": task_pos,
            "agent_pos": agent_pos,
            "total_reward": reward,  # dense + sparse (used for learning curve)
            "task_reward": total_reward,  # sparse completion-only reward
            "rbf_reward": rbf_scaled
        }

        # 检查是否完成
        # Termination: underlying env.done, max_steps, or all tasks cleared
        all_tasks_cleared = (self.current_step > 0) and (not np.any(task_active)) and (len(self.env.accidents) == 0)
        done = self.env.done or all_tasks_cleared or (self.current_step >= self.max_steps)
        if done and all_tasks_cleared:
            # finishing bonus encourages completing all tasks quickly
            reward += 50.0
            info["total_reward"] = reward
            info["finish_bonus"] = 50.0
        else:
            info["finish_bonus"] = 0.0

        return state, reward, done, info

    def render(self):
        """渲染环境"""
        if hasattr(self.env, 'render'):
            self.env.render(self.epoch // 50, self.current_step)