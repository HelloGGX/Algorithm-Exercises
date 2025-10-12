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
    

def find_minimum_cost_network(n, routes):
    # 第一步：排序
    routes.sort(key=lambda x: x[2])

    # 第二、三步：初始化并查集
    uf = UnionFind(n)

    # 第四步：遍历和构建
    mst_routes = []
    total_cost = 0
    num_edges = 0

    for u, v, cost in routes:
        # 如果合并成功（说明之前不连通）
        if uf.union(u, v):
            mst_routes.append([u, v, cost])
            total_cost += cost
            num_edges += 1
            # 优化：当边数等于N-1时，已经构成一棵树，可以提前退出
            if num_edges == n - 1:
                break

    # 如果最终边数少于N-1，说明无法连通所有城市
    if num_edges < n - 1:
        return "无法连通所有城市", -1, []

    return total_cost, mst_routes

# 示例用法
# 假设有4个城市 (0, 1, 2, 3)
n_cities = 4
candidate_routes = [
    [0, 1, 10], [0, 2, 6], [0, 3, 5],
    [1, 3, 15],
    [2, 3, 4]
]

cost, plan = find_minimum_cost_network(n_cities, candidate_routes)

print(f"最低总成本: {cost}")
print("建设方案:")
for route in plan:
    print(f"  连接 {route[0]} 和 {route[1]}, 成本 {route[2]}")
    
    
    
"""
主题：最经济的转机路线

一位旅客希望从 出发城市 (start) 前往 目的城市 (end)，并最多只进行 K 次转机。给定一个航班列表 flights，其中每条记录为 [from, to, price]。

你的任务是，找出满足最多 K 次转机条件下，从 start 到 end 的最低总票价。如果不存在这样的路线，则返回 -1。

示例:
n = 3 (城市数量), flights = [[0,1,100],[1,2,100],[0,2,500]]
start = 0, end = 2, K = 1
输出: 200 (路线 0 -> 1 -> 2)

提示: 这是一个经典的动态规划问题，也与图的最短路径算法（Bellman-Ford）密切相关。
"""
def min_flight_cost_corrected(n, flights, start, end, K):
    """
    状态定义: dp[k][city]
    含义: 从 出发城市(start) 乘坐 恰好 k 次航班 到达 city 的最低总票价。
            city 0, city 1, city 2, city 3   (n=4列)
    k=0: [  inf,    inf,    inf,    inf  ]  <-- 第1行
    k=1: [  inf,    inf,    inf,    inf  ]  <-- 第2行
    k=2: [  inf,    inf,    inf,    inf  ]  <-- 第3行
    """
    # --- 修正点 1: 创建并初始化DP数组 ---
    # K次转机 = 最多 K+1 条航线。DP数组的行数需要 K+2 (索引 0 到 K+1)
    dp = [[float('inf')] * n for _ in range(K + 2)]
    
    # --- 修正点 2: 设置初始条件 (Base Case) ---
    # 乘坐0次航班到达出发城市，成本为0
    dp[0][start] = 0
    
    # --- 你的核心逻辑 (这部分是正确的) ---
    # 遍历航班次数 k，从1到 K+1
    for k in range(1, K + 2):
        # 遍历每一条航线
        for u, v, price in flights:
            # 只有当城市u在k-1步时是可达的，这条航线才有意义
            if dp[k-1][u] != float('inf'):
                # 状态转移方程
                dp[k][v] = min(dp[k][v], dp[k-1][u] + price)
    
    # --- 寻找最终答案 (这部分也是正确的) ---
    # 在dp表格的'end'那一列中寻找最小值 (1次到K+1次航班)
    min_fare = float('inf')
    for k in range(1, K + 2):
        min_fare = min(min_fare, dp[k][end])

    # 如果min_fare还是无穷大，说明不可达
    if min_fare == float('inf'):
        return -1
    else:
        return min_fare

# 示例用法
n = 3
flights = [[0,1,100],[1,2,100],[0,2,500]]
start = 0
end = 2
K = 1

# 调用修正后的函数
cost = min_flight_cost_corrected(n, flights, start, end, K)
print(f"在最多 {K} 次转机条件下，最低票价为: {cost}")
# 期望输出: 200
    
        
  