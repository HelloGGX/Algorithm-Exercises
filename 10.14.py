"""
主题：航站楼最短路径

这个问题我们之前已经定义过，现在请你动手实现它。

假设一个航站楼地图是 m x n 的网格 grid，'S'是起点，'E'是出口，'.'是通道，'#'是墙壁。计算从 'S' 走到 'E' 的最少步数。如果无法到达，返回 -1。

提示: 使用广度优先搜索（BFS），并用一个队列来辅助实现。
"""

from collections import deque

def shortest_path_in_terminal(grid: list[list[str]]) -> int:
    if not grid or not grid[0]:
        return -1
    
    rows, cols = len(grid), len(grid[0])
    
    # --- 第一步：初始化 ---
    queue = deque()
    visited = set()
    
    # 找到起点'S'
    start_pos = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 'S':
                start_pos = (r, c)
                break
        if start_pos:
            break
    
    # 如果没有找到起点
    if not start_pos:
        return -1
    
    queue.append((start_pos[0], start_pos[1], 0))
    visited.add((start_pos[0], start_pos[1]))
    
    # 定义四个移动方向
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] # 右, 左, 下, 上
    
    while queue:
        # 从队头取出一个元素
        r,c,steps = queue.popleft()

        # 检查是否到达终点
        if grid[r][c] == 'E':
            return steps
        
        # --- 第三步：探索邻居 ---
        for dr, dc in directions:
            nr, nc = r+dr, c+dc
            # 对邻居进行合法性检查
            if (0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#' and (nc, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc, steps + 1))
                
    return -1
        



grid = [
  ["S",".",".","#"],
  [".","#",".","#"],
  [".",".",".","."],
  ["#","#","E","#"]
]

steps = shortest_path_in_terminal(grid)
print(f"从'S'到'E'的最少步数是: {steps}") # 期望输出: 7