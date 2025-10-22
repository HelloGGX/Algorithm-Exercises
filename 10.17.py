import collections
import heapq
import re

"""
第一题 (数据结构与算法)

主题：热门航线统计

给定一个航线列表 routes，其中每个元素是形如 "JFK-LAX" 的字符串。你的任务是找出出现频率最高的 k 条航线。

示例: routes = ["JFK-LAX", "JFK-ORD", "SFO-JFK", "JFK-LAX", "SFO-JFK", "JFK-LAX"], k = 2 输出: ["JFK-LAX", "SFO-JFK"] (顺序不重要)

提示: 这个问题考察对哈希表（字典）和堆（优先队列）的综合运用。
"""

def top_k_lines(routes, k):
  route_map = collections.Counter(routes)
  queue = []
  
  for route, count in route_map.items():
    if len(queue) < k:
      heapq.heappush(queue, (count, route))
    elif count > queue[0][0]:
      heapq.heapreplace(queue, (count, route))
  
  return queue
      
routes = ["JFK-LAX", "JFK-ORD", "SFO-JFK", "JFK-LAX", "SFO-JFK", "JFK-LAX"]
res = top_k_lines(routes, 2)
print(res)

"""
第二题 (传统算法 - 动态规划/回溯)

主题：飞行指令解码

你收到一段由无空格字母组成的飞行指令字符串 s，以及一个包含有效指令词的字典 wordDict。你的任务是判断字符串 s 是否可以被空格拆分成一个或多个字典中出现的有效指令词。

示例: s = "flightplan" wordDict = ["flight", "plan"] 输出: True (因为 "flightplan" 可以被拆分为 "flight plan")

s = "applepenapple" wordDict = ["apple", "pen"] 输出: True (因为 "applepenapple" 可以被拆分为 "apple pen apple")

提示: 这是一个经典的“单词拆分”问题，是动态规划或带记忆化搜索的回溯算法的绝佳应用场景。
"""

def word_break_dp(s: str, wordDict: list[str]) -> bool:
    """
    使用动态规划解决单词拆分问题。
    """
    # 为了快速查询，将列表转换为集合
    word_set = set(wordDict)
    n = len(s)
    
    # dp[i] 表示 s 的前 i 个字符组成的子串 s[0:i-1] 是否可以被拆分
    # dp 数组的大小是 n+1，dp[0] 代表空字符串
    dp = [False] * (n + 1)
    dp[0] = True  # Base Case: 空字符串总是可以被“拆分”
    
    # i 代表我们正在尝试填充 dp[i]
    for i in range(1, n + 1):
        # j 代表一个可能的“断点”或“上一块踏脚石”
        for j in range(i):
            # 如果 dp[j] 是 True (s[0:j-1]可拆分)
            # 并且 s[j:i] (从j到i-1的子串) 是一个有效的单词
            if dp[j] and s[j:i] in word_set:
                # 那么 dp[i] 也是 True
                dp[i] = True
                # 只要找到一种拆分方式，就可以停止内层循环，去检查下一个i
                break
                
    # 最终的答案就是 dp[n]，代表整个字符串 s 是否可以被拆分
    return dp[n]

# 示例
s = "catsandog"
wordDict = ["cats", "dog", "sand", "and", "cat"]
print(f"'{s}' 是否可以被拆分? {word_break_dp(s, wordDict)}") # 输出: True
  
"""
第一题 (数据结构 - 30% 分值区)

主题：航班调度指令解析器
你需要编写一个程序，用于解析和执行一系列嵌套的航班调度指令。指令以字符串形式给出，格式类似于数学表达式，包含指令（单个大写字母，代表具体操作）、参数（数字）和括号，用于控制执行优先级。

任务要求：
指令解析: 使用栈的原理，将给定的中缀表达式形式的指令字符串（例如 A(1,B(2,3))）转换为后缀表达式（逆波兰表示法）。
构造执行树: 根据后缀表达式，构建一棵二叉表达式树。在这棵树中，叶子节点是参数（数字），非叶子节点是指令（操作符）。
执行指令: 对构建好的二叉树进行后序遍历，模拟指令的执行并返回最终结果。假设指令 A(x,y) 执行 x+y，指令 B(x,y) 执行 x*y。

示例:
输入指令: "A(5,B(3,4))"
后缀表达式应为: 5 3 4 B A
最终执行结果: 17 (计算过程为 3 * 4 = 12, 然后 5 + 12 = 17)
"""

def flight_command(commands):
  #指令解析
  stack = []
  for i in range(len(commands)):
    if commands[i] != ',':
      stack.append(commands[i])
    while stack and stack[-1] == ')':
      stack.pop()
      y = stack.pop()
      x = stack.pop()
      stack.pop()
      command = stack.pop()
      if command == 'B':
        stack.append(int(x) * int(y))      
      elif command == 'A':
        stack.append(int(x) + int(y))
  
  return stack[0]  
  
res = flight_command("A(5,B(3,4))")
print(res)


"""
标准解法
"""
# 第一步：指令重排 (中缀转后缀)
def infix_to_postfix(command_str: str) -> list:
  """将中缀指令字符串转换为后缀列表。"""
  operators = {'A', 'B'}
  op_stack = []
  postfix_list = []
  
  for token in command_str:
    if token.isdigit():
      postfix_list.append(int(token))
    elif token in operators:
      op_stack.append(token)
    elif token == ')':
      if op_stack:
        postfix_list.append(op_stack.pop())
  return postfix_list
  

# --- 第二步：执行指令 (计算后缀表达式) ---
def evaluate_postfix(postfix_list: list) -> int:
  """根据后缀列表计算最终结果。"""
  eval_stack = []
  # "A(5,B(3,4))" -> [5,3,4,B,A]
  for token in postfix_list:
    if token.isdigit():
      eval_stack.append(token)
    else:
      y = eval_stack.pop()
      x = eval_stack.pop()
      
      if token == 'A':
        eval_stack.append(x + y)
      elif token == 'B':
        eval_stack.append(x * y)
  
  return eval_stack[0]
      
"""
第二题 (传统算法 - 30% 分值区)

主题：枢纽机场网络优化
某区域有 N 个城市，你需要规划一个最低成本的航线网络，并在此基础上分析特定航线。给定一份备选航线列表 routes，每条记录为 [城市A, 城市B, 建设成本]。

任务要求：
最低成本建网: 首先，计算出能够连接所有城市，且总建设成本最低的航线网络方案。输出这个最低的总成本，并列出构成该网络的所有航线。
最短路径分析: 在你上一步构建好的最低成本网络的基础上，计算从指定的 枢纽机场A 到 枢纽机场B 的最短路径（以建设成本为权重）。输出这条最短路径的总成本。

示例:
N=4 个城市, routes = [[0,1,10], [0,2,6], [0,3,5], [1,3,15], [2,3,4]]
枢纽机场A=0, 枢纽机场B=1

输出:
最低总成本: 19。建网方案: [[2,3,4], [0,3,5], [0,1,10]]。
基于该网络，从0到1的最短路径成本: 10。
"""

import heapq

class UnionFind:
  def __init__(self, n) -> None:
    """
    # 链状结构: 0 ← 1 ← 2 ← 3
    # 对应的 parent 数组:
    self.parent = [0, 0, 1, 2]
    # 验证:
    # - 元素0: parent[0] = 0  (0是根节点)
    # - 元素1: parent[1] = 0  (1的父节点是0)
    # - 元素2: parent[2] = 1  (2的父节点是1)  
    # - 元素3: parent[3] = 2  (3的父节点是2)
    """
    self.parent = list(range(n))
  # 压缩路径
  def find(self, i):
    if self.parent[i] != i:
      self.parent[i] = self.find(self.parent[i])
    return self.parent[i]
  
  def union(self, i, j):
    root_i, root_j =  self.find(i), self.find(j)
    if root_i != root_j:
      self.parent[root_i] = root_j
      return True
    return False
  
def min_cost_network(N, routes):
  routes.sort(key=lambda x: x[2])
  
  uf = UnionFind(N)
  
  # 第四步：遍历和构建
  mst_routes = []
  total_cost = 0
  num_edges = 0
  
  for u, v, cost in routes:
    # 如果合并成功，说明之前不联通
    if uf.union(u, v):
      mst_routes.append([u, v, cost])
      total_cost += cost
      num_edges += 1
      # 优化：当边数等于N-1时，已经构成一棵树，可以提前退出
      if num_edges == N - 1:
          break
        
   # 如果最终边数少于N-1，说明无法连通所有城市
  if num_edges < N - 1:
        return "无法连通所有城市", -1, []

  return total_cost, mst_routes

def min_cost_route(N, routes, S, D):
  graph = collections.defaultdict(list)
  for u, v , cost in routes:
    graph[u].append((v, cost))
  
   # distances 字典存储从起点 S 到各城市的已知最短成本
  distances = {i: float('inf') for i in range(N)}
  distances[S] = 0
  
  # 定义优先级队列
  priority_queue = [(0, S)]
  
  # previous_nodes 字典用于回溯最终路径
  previous_nodes = {i: None for i in range(N)}
  
  while priority_queue:
    current_cost, current_airport = heapq.heappop(priority_queue)
    
    if current_cost > distances[current_airport]:
      continue
    
    for neighbor, cost in graph[current_airport]:
      new_cost = cost + distances[current_airport]
      if new_cost < distances[neighbor]:
        distances[neighbor] = new_cost
        previous_nodes[neighbor] = current_airport
        heapq.heappush(priority_queue, (new_cost, neighbor))

  path = []
  total_cost = distances[D]
  
  if total_cost == float('inf'):
    return f"无法从机场 {S} 到达机场 {D}", -1
  else:
    current = D
    while current is not None:
      path.append(current)
      current = previous_nodes[current]
      # 列表反转得到从起点到终点的正确顺序
    path.reverse()
    return path, total_cost
  
# --- 测试 ---
# 使用你之前定义的routes
routes_for_dijkstra = [[0,1,10], [0,2,6], [0,3,5], [1,3,15], [2,3,4]]
# 在你之前构建的MST网络上测试从0到1的最短路径
# MST Edges: [[2,3,4], [0,3,5], [0,1,10]]
mst_network_routes = [[2,3,4], [3,2,4], [0,3,5], [3,0,5], [0,1,10], [1,0,10]] # 假设为双向

path, cost = min_cost_route(4, mst_network_routes, 0, 1)
print(f"从 0 到 1 的最短路径是: {path}, 成本是: {cost}")

