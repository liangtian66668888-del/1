import os
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from env.MultiEnv import MEnv
from env.Entity import Plane, WorkNode, Agent
from env.Route_path import route_Agent
import env.config as config
import random
import pygame
clock = pygame.time.Clock()
class ShipEnv(MEnv):
    def __init__(self, render_mode=None):
        """
        Args:
            render_mode: None表示不创建窗口（训练模式），'human'表示创建窗口（可视化模式）
        """
        pygame.init()
        pygame.display.set_caption("Big Ship Env")
        # 延迟创建窗口，避免在训练时阻塞
        self.render_mode = render_mode
        self.screen = None
        self.clock = pygame.time.Clock()
        self.running = True
        self.UNIT = 30
        self._window_created = False
        # ——缓存字体（使用默认字体作为后备）
        try:
            self.font15 = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 15)
            self.font30 = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 30)
        except:
            # 如果中文字体不存在，使用默认字体
            self.font15 = pygame.font.Font(None, 15)
            self.font30 = pygame.font.Font(None, 30)
        
        # 如果指定了render_mode为'human'，立即创建窗口
        if self.render_mode == 'human':
            self._create_window()

        # ——缓存图片（只load一次）
        current_dir = os.path.dirname(os.path.abspath(__file__))

        ship_path = os.path.join(current_dir, "source", "ship2.jpg")
        self.ship_bg = pygame.transform.scale(pygame.image.load(ship_path), (945, 600))

        accident_path = os.path.join(current_dir, "source", "accident.png")
        self.accident_img = pygame.transform.scale(pygame.image.load(accident_path), (self.UNIT, self.UNIT))

        plane_path = os.path.join(current_dir, "source", "plane2.png")
        self.plane_img = pygame.transform.scale(pygame.image.load(plane_path), (self.UNIT, self.UNIT))

        agent_path = os.path.join(current_dir, "source", "agent4.png")
        self.agent_img = pygame.transform.scale(pygame.image.load(agent_path), (self.UNIT, self.UNIT))
        random.seed()
        self.seed()
        self.observation_space = []  # 观测空间
        self.action_space = []  # 动作空间
        self.nA = 2  # 动作集长度
        # self.obs_dim = (len(config.agent_loc)-1)*2*2+2
        self.obs_dim = (len(config.agent_loc) - 1) * 2
        for _ in range(len(config.agent_loc)):
            u_action_space = spaces.Discrete(self.nA)
            self.action_space.append(u_action_space)
            self.observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32))

        self.GRID_WIDTH = config.GRID_WIDTH
        self.GRID_HEIGHT = config.GRID_HEIGHT
        self.UNIT = config.UNIT
        self.h_offset = config.h_offset
        self.w_offset = config.w_offset
        self.agent_speed = {
            0: [0, 1],
            1: [0, -1],
            2: [-1, 0],
            3: [1, 0]
        }
        self.speed = 10  # 动画播放速度
        self.accRate_all = 0  # 全图事故率
        self.accRate_works = 0  # 保障节点事故率
        self.allWork_num = 0  # 分配保障作业数
        self.noComWork_num = 0  # 未完成保障作业数
        self.workRate = 0  # 任务完成率
        self.Info = {"accRate_all": 0, "accRate_works": 0, "comRate_works": 0}
        self.done = False
        self.obsTask = {}
        self.obs_n = []
        self.reward_n = []
        self.raw_occupancy = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH))
        self.routes_plane_plot = {}
        self.routes_agent_plot = {}
        for agent_id in range(len(config.planes_loc)):
            self.routes_plane_plot[agent_id] = []

        for agent_index in range(len(config.agent_loc)):
            self.routes_agent_plot[agent_index] = []

        # 不可移动区域
        for pos in config.blockArea_loc:
            self.raw_occupancy[pos[0], pos[1]] = -1
        self.occupancy = self.raw_occupancy.copy()

        # 舰载机集合
        self.planes = {}
        for plane_index in range(len(config.planes_loc)):
            angle = -90 if config.planes_loc[plane_index][0] == 0 else 0
            planeHead = 0 if config.planes_loc[plane_index][0] == 0 else 3
            self.planes[plane_index] = Plane(plane_index, config.planes_loc[plane_index], angle, work_id=0,
                                             start_time=plane_index, plane_head=planeHead)
            self.planes[plane_index].work['EndPos'] = config.works_loc[self.planes[plane_index].work['currWork']]
            self.allWork_num += 1
            # route_Agent(self.occupancy, self.planes[plane_index])
            # self.planes[plane_index].isroute = True

        # 保障作业集合
        self.workNodes = {}
        for node_index in range(len(config.works_loc)):
            resourceNum = config.workResources_num[node_index + 1]
            self.workNodes[node_index] = WorkNode(node_index, config.works_loc[node_index], resourceNum * 20)

        # 抢险小车集合
        self.agents = {}
        for agent_index in range(len(config.agent_loc)):
            self.agents[agent_index] = Agent(agent_index, config.agent_loc[agent_index])
            self.obsTask[agent_index] = {}
            for index in range(config.obs_task_len):
                self.obsTask[agent_index][index] = [None]
                self.agents[agent_index].comm[index] = [None]

        # 发生事故
        self.accidents = []
        accident_node_indexs = random.sample(list(map(int, self.workNodes.keys())), 2)
        for acc_index in accident_node_indexs:
            self.accidents.append(self.workNodes[acc_index].pos)
            for agent_index in self.agents.keys():
                self.agents[agent_index].tasks.append(self.workNodes[acc_index].pos)
            self.occupancy[self.workNodes[acc_index].pos[0]][self.workNodes[acc_index].pos[1]] = 1

    def reset(self):
        random.seed()
        self.seed()
        self.accRate_all = 0  # 全图事故率
        self.accRate_works = 0  # 保障节点事故率
        self.allWork_num = 0  # 分配保障作业数
        self.noComWork_num = 0  # 未完成保障作业数
        self.workRate = 0  # 任务完成率
        self.Info = {"accRate_all": 0, "accRate_works": 0, "comRate_works": 0}
        self.done = False
        self.occupancy = self.raw_occupancy.copy()
        self.routes_plane_plot = {}
        self.routes_agent_plot = {}
        for agent_id in range(len(config.planes_loc)):
            self.routes_plane_plot[agent_id] = []
        for agent_index in range(len(config.agent_loc)):
            self.routes_agent_plot[agent_index] = []

        # 舰载机集合
        self.planes = {}
        for plane_index in range(len(config.planes_loc)):
            angle = -90 if config.planes_loc[plane_index][0] == 0 else 0
            planeHead = 0 if config.planes_loc[plane_index][0] == 0 else 3
            self.planes[plane_index] = Plane(plane_index, config.planes_loc[plane_index], angle, work_id=0,
                                             start_time=plane_index, plane_head=planeHead)
            self.planes[plane_index].work['EndPos'] = config.works_loc[self.planes[plane_index].work['currWork']]
            self.allWork_num += 1

        # 保障作业集合
        self.workNodes = {}
        for node_index in range(len(config.works_loc)):
            resourceNum = config.workResources_num[node_index + 1]
            self.workNodes[node_index] = WorkNode(node_index, config.works_loc[node_index], resourceNum * 20)

        # 抢险小车集合
        self.agents = {}
        for agent_index in range(len(config.agent_loc)):
            self.agents[agent_index] = Agent(agent_index, config.agent_loc[agent_index])
            self.obsTask[agent_index] = {}
            for index in range(config.obs_task_len):
                self.obsTask[agent_index][index] = [None]
                self.agents[agent_index].comm[index] = [None]

        # 发生事故
        self.accidents = []
        accident_node_indexs = random.sample(list(map(int, self.workNodes.keys())), 2)
        for acc_index in accident_node_indexs:
            self.accidents.append(self.workNodes[acc_index].pos)
            for agent_index in self.agents.keys():
                self.agents[agent_index].tasks.append(self.workNodes[acc_index].pos)
            self.occupancy[self.workNodes[acc_index].pos[0]][self.workNodes[acc_index].pos[1]] = 1

        # 获取观测值
        self.obs_n = []
        for agent_index in self.agents.keys():
            self.obs_n.append(self.get_obs(agent_index))
        return self.obs_n

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, totalstep, currstep, action_n):
        for work_id in self.workNodes:
            if self.workNodes[work_id].isAcc and self.workNodes[work_id].resourceNum > 0:
                self.workNodes[work_id].resourceNum -= 1

        ##############
        #  事故扩散   #
        ##############
        if currstep % 40 == 0 and currstep != 0:
            next_acc_list = []
            for accident_node in self.accidents:
                if accident_node[0] + 1 >= 0 and accident_node[0] + 1 <= self.GRID_HEIGHT - 1:
                    if self.occupancy[accident_node[0] + 1][accident_node[1]] != 1 and \
                            self.occupancy[accident_node[0] + 1][accident_node[1]] != -1 and \
                            self.occupancy[accident_node[0] + 1][accident_node[1]] != 2:
                        self.occupancy[accident_node[0] + 1][accident_node[1]] = 1
                        if [accident_node[0] + 1, accident_node[1]] not in next_acc_list:
                            next_acc_list.append([accident_node[0] + 1, accident_node[1]])

                if accident_node[0] - 1 >= 0 and accident_node[0] - 1 <= self.GRID_HEIGHT - 1:
                    if self.occupancy[accident_node[0] - 1][accident_node[1]] != 1 and \
                            self.occupancy[accident_node[0] - 1][accident_node[1]] != -1 and \
                            self.occupancy[accident_node[0] - 1][accident_node[1]] != 2:
                        self.occupancy[accident_node[0] - 1][accident_node[1]] = 1
                        if [accident_node[0] - 1, accident_node[1]] not in next_acc_list:
                            next_acc_list.append([accident_node[0] - 1, accident_node[1]])

                if accident_node[1] + 1 >= 0 and accident_node[1] + 1 <= self.GRID_WIDTH - 1:
                    if self.occupancy[accident_node[0]][accident_node[1] + 1] != 1 and \
                            self.occupancy[accident_node[0]][accident_node[1] + 1] != -1 and \
                            self.occupancy[accident_node[0]][accident_node[1] + 1] != 2:
                        self.occupancy[accident_node[0]][accident_node[1] + 1] = 1
                        if [accident_node[0], accident_node[1] + 1] not in next_acc_list:
                            next_acc_list.append([accident_node[0], accident_node[1] + 1])

                if accident_node[1] - 1 >= 0 and accident_node[1] - 1 <= self.GRID_WIDTH - 1:
                    if self.occupancy[accident_node[0]][accident_node[1] - 1] != 1 and \
                            self.occupancy[accident_node[0]][accident_node[1] - 1] != -1 and \
                            self.occupancy[accident_node[0]][accident_node[1] - 1] != 2:
                        self.occupancy[accident_node[0]][accident_node[1] - 1] = 1
                        if [accident_node[0], accident_node[1] - 1] not in next_acc_list:
                            next_acc_list.append([accident_node[0], accident_node[1] - 1])

            for new_acc in next_acc_list:
                self.accidents.append(new_acc)
                for work_index in self.workNodes.keys():
                    if not self.workNodes[work_index].isAcc and \
                            (new_acc[0] == self.workNodes[work_index].pos[0]) and \
                            (new_acc[1] == self.workNodes[work_index].pos[1]):
                        self.workNodes[work_index].isAcc = True
                for agent_index in self.agents.keys():
                    self.agents[agent_index].tasks.append(new_acc)

        ##############
        # agent step #
        ##############
        actions = [[] for _ in range(len(config.agent_loc))]
        for i in range(len(action_n)):
            if (action_n[i][0] >= 0 and action_n[i][0] < 0.125) or (
                    action_n[i][0] >= 0.875 and action_n[i][0] <= 1):
                actions[i] = 0
            if action_n[i][0] >= 0.125 and action_n[i][0] < 0.375:
                actions[i] = 1
            if action_n[i][0] >= 0.375 and action_n[i][0] < 0.625:
                actions[i] = 2
            if action_n[i][0] >= 0.625 and action_n[i][0] < 0.875:
                actions[i] = 3

        self.obs_n = []
        self.reward_n = []
        for agent_index in self.agents.keys():
            if self.agents[agent_index].pos[0] + self.agent_speed[actions[agent_index]][0] >= 0 and \
                    self.agents[agent_index].pos[0] + self.agent_speed[actions[agent_index]][
                0] <= self.GRID_HEIGHT - 1 and self.agents[agent_index].pos[1] + self.agent_speed[actions[agent_index]][
                1] <= self.GRID_WIDTH - 1 and self.agents[agent_index].pos[1] + self.agent_speed[actions[agent_index]][
                1] >= 0:
                if self.occupancy[self.agents[agent_index].pos[0] + self.agent_speed[actions[agent_index]][0]][
                    self.agents[agent_index].pos[1] + self.agent_speed[actions[agent_index]][1]] != -1:
                    self.agents[agent_index].pos = [
                        self.agents[agent_index].pos[0] + self.agent_speed[actions[agent_index]][0],
                        self.agents[agent_index].pos[1] + self.agent_speed[actions[agent_index]][1]
                    ]
            self.routes_agent_plot[agent_index].append(self.agents[agent_index].pos)
            if self.agents[agent_index].pos in self.accidents:
                del self.accidents[
                    self.accidents.index([self.agents[agent_index].pos[0], self.agents[agent_index].pos[1]])
                ]
                del self.agents[agent_index].tasks[
                    self.agents[agent_index].tasks.index([self.agents[agent_index].pos[0],
                                                          self.agents[agent_index].pos[1]])
                ]
                self.occupancy[self.agents[agent_index].pos[0]][self.agents[agent_index].pos[1]] = 2
                self.routes_agent_plot[agent_index] = []

            # 获取观测值
            self.obs_n.append(self.get_obs(agent_index))
            self.reward_n.append(self.reward(agent_index))
        ##############
        # plane step #
        ##############
        for plane_index in self.planes.keys():
            if self.planes[plane_index].first_route and self.planes[plane_index].start_time == currstep:
                route_Agent(self.occupancy, self.planes[plane_index])
                self.routes_plane_plot[plane_index] = []
                self.planes[plane_index].isroute = True
                self.planes[plane_index].first_route = False

            if self.planes[plane_index].route:
                for work_id in self.workNodes.keys():
                    if (self.workNodes[work_id].pos[0] == self.planes[plane_index].pos[0]) \
                            and (self.workNodes[work_id].pos[1] == self.planes[plane_index].pos[1]) \
                            and self.workNodes[work_id].isAcc and not self.planes[plane_index].accRoute and \
                            self.workNodes[work_id].resourceNum == 0:
                        self.planes[plane_index].isroute = False
                        self.planes[plane_index].needAccRoute = True
                        self.noComWork_num += 1
                        workid = 0
                        self.planes[plane_index].work['currWork'] = workid
                        self.planes[plane_index].work['EndPos'] = config.planes_loc[plane_index]

            if not self.planes[plane_index].route and \
                    self.planes[plane_index].work['EndPos'] == self.planes[plane_index].pos:
                self.planes[plane_index].isroute = False
                if self.planes[plane_index].work['currWork'] == len(config.works_loc) - 1 and \
                        self.planes[plane_index].work['EndPos'] != config.planes_loc[plane_index]:
                    self.planes[plane_index].work['EndPos'] = config.planes_loc[plane_index]
                elif (self.planes[plane_index].work['currWork'] == len(config.works_loc) - 1 and \
                      self.planes[plane_index].work['EndPos'] == config.planes_loc[plane_index]) or \
                        (self.planes[plane_index].work['currWork'] == 0 and \
                         self.planes[plane_index].work['EndPos'] == config.planes_loc[plane_index]):
                    # if self.planes[plane_index].work['currWork'] == len(config.works_loc) - 1 and \
                    #         self.planes[plane_index].work['EndPos'] == config.planes_loc[plane_index]:
                    #     self.comWork_num += 1
                    # elif self.planes[plane_index].work['currWork'] == 0 and \
                    #         self.planes[plane_index].work['EndPos'] == config.planes_loc[plane_index]:
                    self.allWork_num += 1
                    workid = 0
                    self.planes[plane_index].work['currWork'] = workid
                    self.planes[plane_index].work['EndPos'] = \
                        config.works_loc[self.planes[plane_index].work['currWork']]
                else:
                    self.planes[plane_index].work['currWork'] += 1
                    self.planes[plane_index].work['EndPos'] = \
                        config.works_loc[self.planes[plane_index].work['currWork']]

            # plane移动
            if self.planes[plane_index].isroute:
                pre_pos = self.planes[plane_index].pos
                self.planes[plane_index].pos = self.planes[plane_index].route.pop()
                self.routes_plane_plot[plane_index].append(self.planes[plane_index].pos)

                if (pre_pos[0] - self.planes[plane_index].pos[0]) > 0 and (
                        pre_pos[1] - self.planes[plane_index].pos[1]) == 0:
                    self.planes[plane_index].pre_plane_head = self.planes[plane_index].plane_head
                    self.planes[plane_index].plane_head = 1
                elif (pre_pos[0] - self.planes[plane_index].pos[0]) <= 0 and (
                        pre_pos[1] - self.planes[plane_index].pos[1]) == 0:
                    self.planes[plane_index].pre_plane_head = self.planes[plane_index].plane_head
                    self.planes[plane_index].plane_head = 0
                elif (pre_pos[0] - self.planes[plane_index].pos[0]) == 0 and (
                        pre_pos[1] - self.planes[plane_index].pos[1]) > 0:
                    self.planes[plane_index].pre_plane_head = self.planes[plane_index].plane_head
                    self.planes[plane_index].plane_head = 2
                elif (pre_pos[0] - self.planes[plane_index].pos[0]) == 0 and (
                        pre_pos[1] - self.planes[plane_index].pos[1]) <= 0:
                    self.planes[plane_index].pre_plane_head = self.planes[plane_index].plane_head
                    self.planes[plane_index].plane_head = 3

                if self.planes[plane_index].pre_plane_head == 0:
                    if self.planes[plane_index].plane_head == 0:
                        self.planes[plane_index].angle = self.planes[plane_index].angle
                    elif self.planes[plane_index].plane_head == 2:
                        self.planes[plane_index].angle = self.planes[plane_index].angle - 90
                    elif self.planes[plane_index].plane_head == 3:
                        self.planes[plane_index].angle = self.planes[plane_index].angle + 90
                    elif self.planes[plane_index].plane_head == 1:
                        self.planes[plane_index].angle = self.planes[plane_index].angle + 180
                elif self.planes[plane_index].pre_plane_head == 1:
                    if self.planes[plane_index].plane_head == 1:
                        self.planes[plane_index].angle = self.planes[plane_index].angle
                    elif self.planes[plane_index].plane_head == 2:
                        self.planes[plane_index].angle = self.planes[plane_index].angle + 90
                    elif self.planes[plane_index].plane_head == 3:
                        self.planes[plane_index].angle = self.planes[plane_index].angle - 90
                    elif self.planes[plane_index].plane_head == 0:
                        self.planes[plane_index].angle = self.planes[plane_index].angle - 180
                elif self.planes[plane_index].pre_plane_head == 2:
                    if self.planes[plane_index].plane_head == 2:
                        self.planes[plane_index].angle = self.planes[plane_index].angle
                    elif self.planes[plane_index].plane_head == 1:
                        self.planes[plane_index].angle = self.planes[plane_index].angle - 90
                    elif self.planes[plane_index].plane_head == 0:
                        self.planes[plane_index].angle = self.planes[plane_index].angle + 90
                    elif self.planes[plane_index].plane_head == 3:
                        self.planes[plane_index].angle = self.planes[plane_index].angle + 180
                elif self.planes[plane_index].pre_plane_head == 3:
                    if self.planes[plane_index].plane_head == 3:
                        self.planes[plane_index].angle = self.planes[plane_index].angle
                    elif self.planes[plane_index].plane_head == 1:
                        self.planes[plane_index].angle = self.planes[plane_index].angle + 90
                    elif self.planes[plane_index].plane_head == 0:
                        self.planes[plane_index].angle = self.planes[plane_index].angle - 90
                    elif self.planes[plane_index].plane_head == 2:
                        self.planes[plane_index].angle = self.planes[plane_index].angle - 180

            elif not self.planes[plane_index].isroute and not self.planes[plane_index].first_route:
                route_Agent(self.occupancy, self.planes[plane_index])
                self.routes_plane_plot[plane_index] = []
                self.planes[plane_index].isroute = True
                if self.planes[plane_index].needAccRoute:
                    self.planes[plane_index].needAccRoute = False
                    self.planes[plane_index].accRoute = True
        accNum = 0

        if not self.accidents:
            self.done = True

        if totalstep == currstep or self.done:
            self.accRate_all = len(self.accidents) / (self.GRID_HEIGHT * self.GRID_WIDTH)
            self.Info["accRate_all"] = self.accRate_all
            for work_index in self.workNodes.keys():
                if self.workNodes[work_index].pos in self.accidents:
                    accNum += 1
            self.accRate_works = accNum / len(self.workNodes)
            self.Info["accRate_works"] = self.accRate_works
            self.workRate = (self.allWork_num - self.noComWork_num) / self.allWork_num
            self.Info["comRate_works"] = self.workRate

        return self.obs_n, self.reward_n, self.done, self.Info

    def get_obs(self, agent_index):
        # 其他智能体和当前智能体的相对距离
        otherAgent = []
        otherAgent_dis = []
        for agentIndex in self.agents.keys():
            if agentIndex != agent_index:
                otherAgent.append(agentIndex)
                otherAgent_dis.append(np.array(
                    [self.agents[agent_index].pos[0] - self.agents[agentIndex].pos[0],
                     self.agents[agent_index].pos[1] - self.agents[agentIndex].pos[1]]
                ))

        # # 任务与agent的相对距离
        for obsTask_index in self.obsTask[agent_index].keys():
            if self.obsTask[agent_index][obsTask_index] not in self.agents[agent_index].tasks:
                taskSpots_list = list(map(list, self.obsTask[agent_index].values()))
                cha_task = []  # 可选任务列表
                for task in self.agents[agent_index].tasks:
                    if task not in taskSpots_list:
                        cha_task.append(task)
                if cha_task:
                    task_index = random.randint(0, len(cha_task) - 1)
                    self.obsTask[agent_index][obsTask_index] = cha_task[task_index]
                    self.agents[agent_index].comm[obsTask_index] = cha_task[task_index]
        # taskToagent_pos = []
        # for index in self.obsTask[agent_index].keys():
        #     if self.obsTask[agent_index][index] != [None]:
        #         taskToagent_pos.append(np.array(
        #             [self.agents[agent_index].pos[0] - self.obsTask[agent_index][index][0],
        #              self.agents[agent_index].pos[1] - self.obsTask[agent_index][index][1]]
        #         ))
        #     else:
        #         taskToagent_pos.append(np.zeros((2,)))

        # 交流信息
        # comm = []
        # for other_index in otherAgent:
        #     for comm_index in self.agents[other_index].comm.keys():
        #         comm.append(self.agents[other_index].comm[comm_index]) if \
        #             self.agents[other_index].comm[comm_index] != [None] else comm.append([0, 0])
        # obs = self.Datastandard(np.concatenate(otherAgent_dis + taskToagent_pos + comm))
        obs = np.concatenate(otherAgent_dis)
        return list(map(int, obs))

    def reward(self, agent_index):
        reward = 0
        for index in self.obsTask[agent_index]:
            if self.obsTask[agent_index][index] != [None]:
                dist = np.square((self.obsTask[agent_index][index][0] - self.agents[agent_index].pos[0])) + \
                       np.square((self.obsTask[agent_index][index][1] - self.agents[agent_index].pos[1]))
                if dist != 0:
                    reward += -dist / 100
                else:
                    reward += 2
        return reward

    def _create_window(self):
        """创建Pygame窗口"""
        if not self._window_created:
            self.screen = pygame.display.set_mode((945, 400))
            self._window_created = True
    
    def render(self, episode, step, close=False):
        if not getattr(self, "running", True):
            return
        
        # 延迟创建窗口，只在需要渲染时创建
        if not self._window_created:
            self._create_window()
        
        if self.screen is None:
            return

        # 1) 处理事件（必须）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

        # 2) 控帧
        self.clock.tick(self.speed)

        # 3) 清屏 + 背景
        self.screen.fill((220, 220, 220))
        self.screen.blit(self.ship_bg, (0, -110))

        # 4) UI文字
        self.screen.blit(self.font15.render(f"Episode: {episode}", True, (0, 0, 0)), (30, 10))
        self.screen.blit(self.font15.render(f"Step: {step}", True, (0, 0, 0)), (140, 10))

        # 5) 保障作业框
        for node_index, node in self.workNodes.items():
            pygame.draw.rect(
                self.screen, (124, 252, 0),
                [node.pos[1] * self.UNIT + self.w_offset,
                 node.pos[0] * self.UNIT + self.h_offset,
                 self.UNIT, self.UNIT],
                1
            )

        # 6) 事故
        for acc in self.accidents:
            self.screen.blit(
                self.accident_img,
                (acc[1] * self.UNIT + self.w_offset, acc[0] * self.UNIT + self.h_offset)
            )

        # 7) 保障作业编号
        for node_index, node in self.workNodes.items():
            self.screen.blit(
                self.font30.render(str(node_index + 1), True, (0, 0, 0)),
                (node.pos[1] * self.UNIT + self.w_offset,
                 node.pos[0] * self.UNIT + self.h_offset)
            )

        # 8) 舰载机
        for plane_index, plane in self.planes.items():
            plane_img_t = pygame.transform.rotate(self.plane_img, plane.angle)
            self.screen.blit(
                plane_img_t,
                (plane.pos[1] * self.UNIT + self.w_offset, plane.pos[0] * self.UNIT + self.h_offset)
            )
            pygame.draw.rect(
                self.screen, (160, 32, 240),
                [plane.pos[1] * self.UNIT + self.w_offset,
                 plane.pos[0] * self.UNIT + self.h_offset,
                 self.UNIT, self.UNIT],
                1
            )

        # 9) 救援agent
        for agent_index, agent in self.agents.items():
            self.screen.blit(
                self.agent_img,
                (agent.pos[1] * self.UNIT + self.w_offset,
                 agent.pos[0] * self.UNIT + self.h_offset)
            )

        # 10) 最后刷新
        pygame.display.flip()

    def close(self):
        if getattr(self, "running", True):
            self.running = False
        pygame.quit()

    @staticmethod
    def Datastandard(data):
        Data_stn = []
        # for i in data:
        #     if (float(i) - np.min(x)) != 0 and np.max(data) - np.min(data) != 0:
        #         list.append((float(i) - np.min(data)) / float(np.max(data) - np.min(data)))
        #     else:
        #         list.append(0)
        # return list
        mean = np.mean(data)
        std = np.std(data, ddof=0)

        for data_index in range(len(data)):
            Data_stn.append((data[data_index] - mean) / std)
        return Data_stn


if __name__ == "__main__":
    env = ShipEnv()
    i = 0
    while i <= 20 and env.running:
        env.reset()
        j = 0
        while j <= 200 and env.running:
            actions = [[random.random(), 0] for _ in range(len(config.agent_loc))]
            env.step(totalstep=200, currstep=j, action_n=actions)
            env.render(i, j)
            j += 1
        i += 1

    env.close()

