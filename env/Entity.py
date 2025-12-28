# 舰载机
class Plane(object):
    def __init__(self, id, pos, angle, work_id, start_time, plane_head):
        self.id = id
        self.pos = pos
        self.angle = angle  # render时机头角度
        self.work = {'currWork': work_id, 'EndPos': []}
        self.route = []
        self.needAccRoute = False
        self.accRoute = False
        self.isroute = False
        self.first_route = True  # 第一次规划路径
        self.start_time = start_time * 20
        self.pre_plane_head = plane_head
        self.plane_head = plane_head


# 抢险小车
class Agent(object):
    def __init__(self, id, pos):
        self.id = id
        self.pos = pos
        self.comm = {}
        self.tasks = []


# 保障作业节点
class WorkNode(object):
    def __init__(self, id, pos, resourceNum):
        self.id = id
        self.pos = pos
        self.resourceNum = resourceNum
        self.isAcc = False  # 保障点是否发生事故
