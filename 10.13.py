"""
今日主题: 图论 (Graph Theory)
题目：航站楼连通性分析
一个大型国际机场的停机坪可以看作是一个 m x n 的网格 grid。
每个单元格可以是停机位（1）或滑行道（0）。一组相邻（上、下、左、右）的停机位 1 构成一个“停机区”。
你的任务是计算出这个机场共有多少个独立的停机区。
示例:
grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出: 3
"""
def numIslands(grid: list[list[str]]) -> int:
  if not grid:
    return 0
  rows, cols = len(grid), len(grid[0])
  count = 0
  
  def dfs(r,c):
    if not (0 <= r < rows and 0 <= c < cols and grid[r][c] == '1'):
      return
    grid[r][c] = '0'
    
    # 向四个方向探索
    dfs(r + 1, c)
    dfs(r - 1, c)
    dfs(r, c + 1)
    dfs(r, c - 1)
  
  for r in range(rows):
    for c in range(cols):
      if grid[r][c] == '1':
        count += 1
        dfs(r, c)
        
  return count