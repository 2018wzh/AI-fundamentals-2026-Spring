"""
待补充代码：对搜索过的格子染色
"""

import matplotlib.pyplot as plt
import heapq
from collections import deque


def visualize_maze_with_path(maze, path=None, visited=None):
    """可视化迷宫。
    maze: 二维列表，0 表示空格，1 表示墙
    path: 列表或元组的坐标序列 e.g. [(x0,y0), ...]
    visited: 列表或集合的坐标，表示搜索过程中访问过的格子（用于染色）
    """
    plt.figure(figsize=(len(maze[0]), len(maze)))  # 设置图形大小
    plt.imshow(maze, cmap="Greys", interpolation="nearest")  # 使用灰度色图，并关闭插值

    # 绘制已访问格子（如果有），使用半透明的方块标记
    if visited:
        try:
            vx, vy = zip(*visited)
            plt.scatter(
                vy, vx, c="cyan", s=200, marker="s", edgecolors="none", alpha=0.6
            )
        except Exception:
            # 如果 visited 为空或格式不对，跳过
            pass

    # 绘制路径
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker="o", markersize=8, color="red", linewidth=3)

    # 设置坐标轴刻度和边框
    plt.xticks(range(len(maze[0])))
    plt.yticks(range(len(maze)))
    plt.gca().set_xticks([x - 0.5 for x in range(1, len(maze[0]))], minor=True)
    plt.gca().set_yticks([y - 0.5 for y in range(1, len(maze))], minor=True)
    plt.grid(which="minor", color="black", linestyle="-", linewidth=2)

    plt.axis("on")  # 显示坐标轴
    plt.show()


# 提供迷宫的二维数组
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
]

# 假设给定路径的坐标列表
path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 4), (4, 4)]

mov = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def dfs(maze):
    """深度优先搜索，返回 (path, visited)
    path: 从起点到终点的路径（若存在），visited: 按访问顺序的格子列表
    """
    rows, cols = len(maze), len(maze[0])
    target = (rows - 1, cols - 1)
    if maze[0][0] == 1 or maze[target[0]][target[1]] == 1:
        return [], []

    visited = []
    visited_flag = [[False] * cols for _ in range(rows)]
    path = []

    def search(x, y):
        if not (0 <= x < rows and 0 <= y < cols):
            return False
        if visited_flag[x][y] or maze[x][y] == 1:
            return False
        visited_flag[x][y] = True
        visited.append((x, y))
        path.append((x, y))
        if (x, y) == target:
            return True
        for dx, dy in mov:
            nx, ny = x + dx, y + dy
            if search(nx, ny):
                return True
        path.pop()
        return False

    search(0, 0)
    return path, visited


def bfs(maze):
    """广度优先搜索，返回 (path, visited)"""
    rows, cols = len(maze), len(maze[0])
    target = (rows - 1, cols - 1)
    if maze[0][0] == 1 or maze[target[0]][target[1]] == 1:
        return [], []

    visited = []
    visited_flag = [[False] * cols for _ in range(rows)]
    parent = {}
    q = deque()
    q.append((0, 0))
    visited_flag[0][0] = True
    visited.append((0, 0))
    parent[(0, 0)] = None

    while q:
        x, y = q.popleft()
        if (x, y) == target:
            # 重建路径
            path = []
            cur = target
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path, visited
        for dx, dy in mov:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < rows
                and 0 <= ny < cols
                and not visited_flag[nx][ny]
                and maze[nx][ny] == 0
            ):
                visited_flag[nx][ny] = True
                parent[(nx, ny)] = (x, y)
                visited.append((nx, ny))
                q.append((nx, ny))
    return [], visited


def dijkstra(maze):
    """Dijkstra（适用于加权图），这里网格权重为1。返回 (path, visited)"""
    rows, cols = len(maze), len(maze[0])
    target = (rows - 1, cols - 1)
    if maze[0][0] == 1 or maze[target[0]][target[1]] == 1:
        return [], []

    dist = [[float("inf")] * cols for _ in range(rows)]
    parent = {}
    visited = []
    heap = []
    dist[0][0] = 0
    heapq.heappush(heap, (0, 0, 0))  # (distance, x, y)

    while heap:
        d, x, y = heapq.heappop(heap)
        if d > dist[x][y]:
            continue
        visited.append((x, y))
        if (x, y) == target:
            # 重建路径
            path = []
            cur = target
            while cur in parent:
                path.append(cur)
                cur = parent[cur]
            path.append((0, 0))
            path.reverse()
            return path, visited
        for dx, dy in mov:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0:
                nd = d + 1
                if nd < dist[nx][ny]:
                    dist[nx][ny] = nd
                    parent[(nx, ny)] = (x, y)
                    heapq.heappush(heap, (nd, nx, ny))
    return [], visited


def astar(maze):
    """A* 搜索，返回 (path, visited)"""
    rows, cols = len(maze), len(maze[0])
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    if maze[0][0] == 1 or maze[goal[0]][goal[1]] == 1:
        return [], []

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap = []
    g_score = [[float("inf")] * cols for _ in range(rows)]
    parent = {}
    visited = []

    g_score[0][0] = 0
    heapq.heappush(open_heap, (heuristic(start, goal), 0, 0, 0))  # (f, g, x, y)

    while open_heap:
        f, g, x, y = heapq.heappop(open_heap)
        if g > g_score[x][y]:
            continue
        visited.append((x, y))
        if (x, y) == goal:
            # 重建路径
            path = []
            cur = goal
            while cur in parent:
                path.append(cur)
                cur = parent[cur]
            path.append(start)
            path.reverse()
            return path, visited
        for dx, dy in mov:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0:
                tentative_g = g + 1
                if tentative_g < g_score[nx][ny]:
                    g_score[nx][ny] = tentative_g
                    parent[(nx, ny)] = (x, y)
                    f2 = tentative_g + heuristic((nx, ny), goal)
                    heapq.heappush(open_heap, (f2, tentative_g, nx, ny))
    return [], visited


def main():
    dfs_path, dfs_visited = dfs(maze)
    visualize_maze_with_path(maze, path=dfs_path, visited=dfs_visited)

    bfs_path, bfs_visited = bfs(maze)
    visualize_maze_with_path(maze, path=bfs_path, visited=bfs_visited)

    dijkstra_path, dijkstra_visited = dijkstra(maze)
    visualize_maze_with_path(maze, path=dijkstra_path, visited=dijkstra_visited)

    astar_path, astar_visited = astar(maze)
    visualize_maze_with_path(maze, path=astar_path, visited=astar_visited)


if __name__ == "__main__":
    main()
