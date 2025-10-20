"""
主题：航站楼最短路径

这个问题我们之前已经定义过，现在请你动手实现它。
假设一个航站楼地图是 m x n 的网格 grid，'S'是起点，'E'是出口，'.'是通道，'#'是墙壁。计算从 'S' 走到 'E' 的最少步数。
如果无法到达，返回 -1。
"""
from collections import deque
from typing import Optional
def shortest_path_in_terminal(grid: list[list[str]]) -> int:
  if not grid or not grid[0]:
    return -1
  rows, cols = len(grid), len(grid[0])
  queue = deque()
  visited = set()
  start_pos = None
  
  for r in range(rows):
    for c in range(cols):
      if grid[r][c] == 'S':
        start_pos = (r, c, 0)
        break
  
  if not start_pos:
    return -1
  
  queue.append(start_pos)
  visited.add((start_pos[0], start_pos[1]))
  
  directions = [(0, 1), (0, -1), (1,0), (-1,0)]
  
  while queue:
    cr, cc, steps = queue.popleft()
    if grid[cr][cc] == 'E':
      return steps
    
    for nr, nc in directions:
      fr = cr+nr
      fc = cc+nc
      if 0<= fr < rows and 0<= fc < cols and grid[fr][fc]!= '#' and (fr, fc) not in  visited:
        queue.append((fr, fc, steps + 1))
        visited.add((fr, fc))
  
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



import torch
import torch.nn as nn

vocab_size = 1000
embedding_dim = 128
hidden_dim = 256
output_size = 2

class SentimentLSTM(nn.Module):
    def __init__(self):
        super(SentimentLSTM, self).__init__()
        # 1. 词典层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 2. 记忆引擎层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 3. 决策层
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out

# --- 测试你的模型 ---
model = SentimentLSTM()
dummy_input = torch.randint(0, vocab_size, (5, 10))
output = model(dummy_input)
print("模型输出形状:", output.shape) # 期望: torch.Size([5, 2])
print("模型输出示例:", output)


"""
第二题 (树与递归)

主题：航线网络合法性校验
航线网络规划中，有时会将航点组织成二叉树结构以便快速查找。
一个合法的“航点查找树”必须是一个有效的二叉搜索树 (BST)。
你的任务是：编写一个函数，判断给定的一个二叉树是否为有效的二叉搜索树。

BST定义:

节点的左子树只包含小于当前节点的数。
节点的右子树只包含大于当前节点的数。
所有左、右子树也必须是二叉搜索树。
提示: 需要在递归中传递上界和下界信息，考察对递归状态的深入理解。
"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
        
class Solution:
  def isValidBST(self, root: Optional[TreeNode]) -> bool:
    return self._is_valid_helper(root, float('-inf'), float('inf'))
  
  def _is_valid_helper(self, node: Optional[TreeNode], lower:float, upper: float) -> bool:
    if not node:
      return True
    
    if not (lower < node.val < upper):
        return False
    
    return self._is_valid_helper(node.left, lower, node.val) and self._is_valid_helper(node.right, node.val, upper)
    