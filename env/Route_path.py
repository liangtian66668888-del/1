from collections import deque
import env.config as config
import numpy as np


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


def route_Agent(maze, agent):
    begin = Point()
    end = Point()
    begin.x = list(map(int, [agent.pos[0]]))[0]
    begin.y = list(map(int, [agent.pos[1]]))[0]
    end.x = agent.work['EndPos'][0]
    end.y = agent.work['EndPos'][1]
    n, m = config.GRID_HEIGHT, config.GRID_WIDTH
    dist = [[np.inf for _ in range(m)] for _ in range(n)]
    pre = [[None for _ in range(m)] for _ in range(n)]  # 当前点的上一个点,用于输出路径轨迹

    dx = [1, 0, -1, 0]  # 四个方位
    dy = [0, 1, 0, -1]
    sx, sy = list(map(int, [begin.x]))[0], list(map(int, [begin.y]))[0]
    gx, gy = list(map(int, [end.x]))[0], list(map(int, [end.y]))[0]

    dist[sx][sy] = 0
    queue = deque()
    queue.append(begin)
    while queue:
        curr = queue.popleft()
        curr.x, curr.y = list(map(int, [curr.x]))[0], list(map(int, [curr.y]))[0]
        find = False
        for i in range(4):
            nx, ny = curr.x + dx[i], curr.y + dy[i]
            if 0 <= nx < n and 0 <= ny < m and maze[nx][ny] != -1 and dist[nx][ny] == np.inf:
                dist[nx][ny] = dist[curr.x][curr.y] + 1
                pre[nx][ny] = curr
                queue.append(Point(nx, ny))
                if nx == gx and ny == gy:
                    find = True
                    break
        if find:
            break

    route_stack = []
    curr = end
    while True:
        route_stack.append([curr.x, curr.y])
        if curr.x == begin.x and curr.y == begin.y:
            break
        prev = pre[curr.x][curr.y]
        curr = prev
    agent.route = route_stack
