"""
主题：航站楼最短路径

这个问题我们之前已经定义过，现在请你动手实现它。
假设一个航站楼地图是 m x n 的网格 grid，'S'是起点，'E'是出口，'.'是通道，'#'是墙壁。计算从 'S' 走到 'E' 的最少步数。
如果无法到达，返回 -1。
"""
from collections import deque
def shortest_path_in_terminal(grid: list[list[str]]) -> int:
  if not grid or not grid[0]:
    return -1
  
  start_po = None
  queue = deque()
  visited = set()
  
  # 1. 找到起点
  rows, cols = len(grid), len(grid[0])
  for r in range(rows):
    for c in range(cols):
      if grid[r][c] == 'S':
        start_po = (r, c)
        break
    if start_po:
      break
  
  if not start_po:
    return -1
  
  queue.append((start_po[0], start_po[1], 0))
  visited.add((start_po[0], start_po[1]))
  
  directions = [(0,1), (0,-1), (1, 0), (-1,0)]
  
  while queue:
    r, c, steps = queue.popleft()
    if grid[r][c] == 'E':
      return steps
    for dr, dc in directions:
      nr, nc = r + dr, c + dc
      if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#' and (nr, nc) not in visited:
        queue.append((nr, nc, steps+1))
        visited.add((nr, nc))
        
  return -1
        

# 示例
grid = [
  ["S",".",".","#"],
  [".","#",".","#"],
  [".",".",".","."],
  ["#","#","E","#"]
]
steps = shortest_path_in_terminal(grid)
print(f"从'S'到'E'的最少步数是: {steps}") # 期望输出: 7