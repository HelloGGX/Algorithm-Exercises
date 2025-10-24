"""
第一题：机场登机口动态分配 (问题建模与模拟 - 红灯区 🚨)

主题: 模拟一个简化的机场登机口分配系统。

机场有 G 个登机口，编号 0 到 G-1。N 架飞机按顺序请求降落，信息存储在列表 flights 中，每个元素是 [request_time, required_gate_time]，代表飞机的请求时间和所需的登机口占用时长。

分配规则如下：

当一架飞机在 request_time 请求时，系统会查找当前空闲的登机口。

如果有多个登机口空闲，优先分配给编号最小的那个。

如果没有登机口空闲，飞机需要排队等待，直到有登机口空闲为止。等待的飞机中，请求时间最早的优先获得下一个空闲的登机口。

登机口一旦被分配，会一直被占用 required_gate_time 这么长的时间。

任务要求： 编写一个程序，模拟这个分配过程，并计算出所有飞机完成登机口占用（即离开登机口）的最晚时间点。

示例:

G = 1 (1个登机口)

flights = [[0, 5], [1, 3], [3, 4]] (飞机0在t=0请求用5单位时间, 飞机1在t=1请求用3, 飞机2在t=3请求用4)

输出: 12

过程解释:

t=0: 飞机0请求，登机口0空闲。分配给飞机0。登机口0将在 0 + 5 = 5 时刻空闲。

t=1: 飞机1请求，登机口0被占用。飞机1进入等待队列。

t=3: 飞机2请求，登机口0仍被占用。飞机2进入等待队列。

t=5: 登机口0空闲。等待队列中有飞机1(t=1请求)和飞机2(t=3请求)。优先分配给请求更早的飞机1。飞机1开始使用登机口。登机口0将在 5 + 3 = 8 时刻空闲。

t=8: 登机口0空闲。等待队列中只有飞机2。分配给飞机2。飞机2开始使用登机口。登机口0将在 8 + 4 = 12 时刻空闲。

t=12: 所有飞机都已完成。最晚完成时间是12。

考纲对应: 队列/堆应用，资源排队，调度模拟。 训练目标: 强制运用“五步建模法”，精确模拟时间和多个资源（登机口）的状态。你需要追踪每个登机口的空闲时间，以及一个等待队列（用什么数据结构最高效？）。
"""

import heapq
from collections import defaultdict 

def gate_assign(G:int, flights:list[list[int]]) -> int:
  
  if not flights:
    return 0
  
  flights.sort(key=lambda x: x[0])
  gates = defaultdict()
  
  # [(request_time, required_gate_time)]
  on_wait = flights
  heapq.heapify(on_wait)
  
  deliver_flight = 0
  n = len(flights)
  
  while deliver_flight < n:
    
    for i in range(G):
      request_time, required_gate_time = heapq.heappop(on_wait)
      # 如果有通道空闲
      if not gates[i]:
        gates[i] = (request_time, required_gate_time)
        deliver_flight += 1
    
    
  
  
  
  
  

  
      
         
         
       
       
    
  
  
    
 
  
    