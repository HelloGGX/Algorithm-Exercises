"""
第一题 (数据结构 - 30% 分值区)
主题：航班延误时长分析
气象部门提供了一份未来 N 小时的连续风力预测数据，以整数数组 wind_speeds 表示。
作为航班调度员，你需要分析一段连续的风力增强区间。
对于每个小时 i，你需要计算出在它之前（包括它自己）有多少个连续的小时，其风速小于或等于 wind_speeds[i]。
示例:
wind_speeds = [100, 80, 60, 70, 60, 75, 85]
输出: [1, 1, 1, 2, 1, 4, 6]
"""

# 单调栈寻找第一个比当前元素大或者小的元素的场景
def wind_speed_analysis(wind_speeds):
  if not wind_speeds:
    return []
    
  n = len(wind_speeds)
  res = [0] * n
  # 栈中存储 (索引, 风速) 的元组
  stack = [] 

  # 从0开始遍历，逻辑更统一
  for i, cur_wind in enumerate(wind_speeds):
    # 核心修正：
    # 1. 检查栈不为空 (while stack and ...)
    # 2. 条件是 <= 而不是 <
    while stack and stack[-1][1] <= cur_wind:
      stack.pop()
    
    # 计算结果
    if not stack:
      # 如果栈为空，说明左侧没有比当前更大的元素
      res[i] = i + 1
    else:
      # 栈顶元素的索引就是左侧第一个更大元素的位置
      prev_greater_idx = stack[-1][0]
      res[i] = i - prev_greater_idx
      
    # 将当前元素压入栈中
    stack.append((i, cur_wind))
    
  return res

wind_speeds = [100, 80, 60, 70, 60, 75, 85]
res = wind_speed_analysis(wind_speeds)
print(res)

"""
主题：机场网络建设规划
为提升区域连通性，计划在一系列尚未通航的城市之间建立新的航线网络。
你拿到了一份包含 N 个城市和 M 条备选航线方案的列表 routes。每条方案 [city1, city2, cost] 代表在 city1 和 city2 之间建立直航航线的建设成本。
你的任务是，设计一个航线建设计划，确保所有 N 个城市最终都能互相连通（直接或间接），同时总的建设成本最低。

要求:

输出这个最低的总建设成本。
输出你选择建设的航线列表。
考纲对应: “最小生成树：Kruskal/Prim算法实现并输出边集合与总权重”。
"""
import collections
def flight_kruskal(N, routes):
  graph = collections.defaultdict(list)