"""
主题：行李安检通道
机场有 K 个行李安检通道。N 件行李按顺序到达安检区，每件行李的处理时间已知，存储在列表 processing_times 中。

行李分配规则如下：
当一件行李到达时，系统会检查所有 K 个通道。
行李会被分配给当前累计处理时间最短的那个通道。如果有多个通道累计处理时间相同，则分配给编号最小的通道。
任务要求： 编写一个程序，模拟这个分配过程，并计算出所有行李都被处理完毕所需的总时间。总时间取决于最后一个完成处理的那个通道的时间。

示例:
K = 2 (2个通道)
processing_times = [5, 1, 3, 2]
输出: 6

过程解释:
行李0 (耗时5) 到达，分配给通道0 (累计时间0最短)。通道0完成时间变为 5。
行李1 (耗时1) 到达，分配给通道1 (累计时间0最短)。通道1完成时间变为 1。
行李2 (耗时3) 到达，通道1 (累计时间1) 比通道0 (累计时间5) 短。分配给通道1。通道1完成时间变为 1 + 3 = 4。
行李3 (耗时2) 到达，通道1 (累计时间4) 比通道0 (累计时间5) 短。分配给通道1。通道1完成时间变为 4 + 2 = 6。
所有行李分配完毕。通道0在时间5完成，通道1在时间6完成。总时间取决于最晚的那个，即 6。
"""
from collections import defaultdict
import heapq

import heapq

def baggage_scheduler(K: int, processing_times: list[int]) -> int:
    """
    使用最小堆精确模拟K个通道的行李分配过程。
    """
    if not processing_times:
        return 0
        
    # 1. 初始化优先队列（最小堆）
    # 堆中存储元组 (通道完成时间, 通道索引)
    channel_heap = [(0, i) for i in range(K)]
    heapq.heapify(channel_heap) # 确保初始堆结构正确
    
    # 记录每个通道最终的完成时间，或者也可以在循环结束后直接取堆中最大值
    # final_channel_times = [0] * K 
    max_finish_time = 0

    # 2. 遍历所有行李
    for i in range(len(processing_times)):
        current_processing_time = processing_times[i]
        
        # a) 从堆中获取当前最快完成的通道
        earliest_finish_time, channel_idx = heapq.heappop(channel_heap)

        # b) 计算该通道处理完此行李的新完成时间
        new_finish_time = earliest_finish_time + current_processing_time
        
        # c) 将更新后的通道信息放回堆中
        heapq.heappush(channel_heap, (new_finish_time, channel_idx))
        
        # d) 更新全局最大完成时间 (或者循环结束后再找最大值)
        max_finish_time = max(max_finish_time, new_finish_time)
    
    return max_finish_time

   

# 示例
K = 2
processing_times = [5, 1, 3, 2]
print(f"总时间: {baggage_scheduler(K, processing_times)}") # 输出: 6
   
processing_times = [5, 1, 3, 2, 1,9,7]
res = baggage_scheduler(3, processing_times)
print(res)
  

"""
主题：航线网络升级

你负责升级一个现有的航线网络。该网络有 N 个机场，当前已有 M 条双向航线，但网络可能不是完全连通的。现在有一份新的备选航线清单 new_routes，每条记录为 [机场A, 机场B, 升级成本]。

任务要求： 计算出最少需要花费多少额外的升级成本，才能使得所有 N 个机场都互相连通。如果即使加上所有新航线也无法连通所有机场，则返回 -1。

示例:

N=4 机场

existing_routes = [[0, 1], [1, 2]] (当前已有的航线，成本视为0)

new_routes = [[0, 2, 100], [2, 3, 200], [0, 3, 500]]

输出: 200 解释: 当前 0-1-2 已连通。为了连接机场3，可以选择成本为200的新航线 [2, 3]。
"""
class UnionFind:
  def __init__(self, n) -> None:
    self.parent = list(range(n))
  def find(self, i):
    if self.parent[i] != i:
      self.parent[i] = self.find(self.parent[i])
    return self.parent[i]
  def union(self, i, j):
    root_i = self.find(i)
    root_j = self.find(j)
    if root_i != root_j:
      self.parent[root_i] = root_j
      return True
    return False
         

def flight_update(N:int, existing_routes:list[list[int]], new_routes:list[list[int]]) -> int:
  existing_routes = [[existing_routes[i][0], existing_routes[i][1], 0] for i in range(len(existing_routes))]
  routes = existing_routes + new_routes
  
  # 重要!，从小到大排序
  routes.sort(key = lambda x: x[2])
  
  n = len(routes)
  num_edges = 0
  result_cost = 0
  
  uf = UnionFind(N)
  
  for u, v, cost in routes:
    # 如果为True说明之前没有连接
    if uf.union(u,v):
      result_cost +=cost
      num_edges += 1
      if num_edges == N - 1:
        break
  
  #如果最终边数少于N-1，说明无法连通所有城市
  if num_edges < N - 1:
    return -1
  
  return result_cost
    
res = flight_update(4, [[0, 1], [1, 2]], [[0, 2, 100], [2, 3, 200], [0, 3, 500]])

print(res)    
  
  
  
  
  
    
  
  

