"""
第一题 (数据结构与问题建模)

主题：航班行李传送带

机场的行李传送带系统可以看作一个循环队列。传送带上共有 N 个位置（编号0到N-1）。行李在传送带上按编号顺序移动，到达位置 N-1 后下一个位置是 0。

现在，有一批行李，每个行李上有一个标签 (start_pos, target_pos)，表示它从哪个位置上带，要到哪个位置被取走。假设传送带每秒移动一个位置。

任务要求： 编写一个程序，模拟这个过程，并计算出运送完所有行李所需的总时间。

传送带初始为空，时间从0开始。

在每个时间点，先检查当前位置是否有行李需要被取下，然后再将新的行李放上。

示例:

N = 5 (传送带共5个位置)

baggage = [(0, 2), (1, 3)]

输出: 4

过程解释:

t=0: 传送带在位置0。行李(0,2)被放上。

t=1: 传送带在位置1。行李(1,3)被放上。

t=2: 传送带在位置2。行李(0,2)被取下。

t=3: 传送带在位置3。行李(1,3)被取下。所有行李运送完毕。

t=4: 时间在所有行李被取下的那一刻结束，总耗时为4秒。
"""

from collections import defaultdict, deque

def baggage_schedule_final(N: int, baggage: list[tuple]) -> int:
    if not baggage:
        return 0

    # 1. 准备状态
    waiting_bags = defaultdict(list)
    for start, target in baggage:
        waiting_bags[start].append(target)
    
    # 这个列表追踪所有在传送带上的行李和它们的当前位置
    on_belt = [] 
    
    total_bags = len(baggage)
    delivered_count = 0
    time = 0

    # 2. 基于时间的主循环
    while delivered_count < total_bags:
        # a) 移动所有在带上的行李
        for i, bag in enumerate(on_belt):
            # 每个行李的位置+1，并循环
            current_station_pos, target = bag
            on_belt[i] = ((current_station_pos + 1) % N, target)

        # b) 取下到站的行李
        remaining_on_belt = []
        for current_pos, target_pos in on_belt:
            if current_pos == target_pos:
                delivered_count += 1
            else:
                remaining_on_belt.append((current_pos, target_pos))
        on_belt = remaining_on_belt
        
        # c) 放上等待的行李
        current_station_pos = time % N
        if current_station_pos in waiting_bags:
            for target in waiting_bags[current_station_pos]:
                on_belt.append((current_station_pos, target))
            del waiting_bags[current_station_pos]

        # d) 时间流逝
        time += 1
        
    return time

# 示例
N = 5
baggage = [(0, 2), (1, 3)]
print(f"运送完所有行李需要的时间: {baggage_schedule_final(N, baggage)}") # 输出: 4

N_complex = 3
baggage_complex = [(0, 1), (0, 2)]
print(f"复杂情况所需时间: {baggage_schedule_final(N_complex, baggage_complex)}") # 输出: 3
# 解释: t=0, pos=0, bag(0,1)和(0,2)上带; t=1, pos=1, bag(0,1)下带; t=2, pos=2, bag(0,2)下带; time=3.
         

 
"""
一架飞机需要执行一段长途飞行任务，起点油箱是满的，可以飞行 max_fuel 公里。沿途有多个加油站，信息存储在列表 stations 中，每个元素是 [distance_from_start, fuel_capacity]。

你的任务是：计算飞机从起点到达终点 target_distance 所需的最少加油次数。如果无法到达，则返回 -1。你可以假设在任何加油站加油都会把油箱加满。

示例:

target_distance = 100, max_fuel = 10

stations = [[10, 60], [20, 30], [30, 30], [60, 40]]

输出: 2

过程解释:

从起点出发，油箱能飞10公里，到达加油站[10, 60]。

在这里加油（第1次），油箱加满，能飞到 10 + 60 = 70 公里处。

在能到达的范围内（70公里内），选择下一个能让你飞得最远的加油站。在[20,30], [30,30], [60,40]中，显然在[60,40]加油最划算。

飞到[60,40]加油（第2次），油箱加满，能飞到 60 + 40 = 100 公里处，正好到达终点。
"""
import heapq
def min_refuel_stops(target: int, start_fuel: int, stations: list[list[int]]) -> int:
  if start_fuel >= target:
        return 0
  
  current_reach = start_fuel
  pq = []
  station_idx = 0
  stops = 0
  
  while current_reach < target:
    while station_idx < len(stations) and stations[station_idx][0] <= current_reach:
      heapq.heappush(pq, -stations[station_idx][1])
      station_idx +=1
    
    if not pq:
      return -1
        
    fuel_from_best_station = -heapq.heappop(pq)
    
    current_reach += fuel_from_best_station
    stops +=1
     
  return stops

# 示例
target_distance = 100
max_fuel = 10
stations = [[10, 60], [20, 30], [30, 30], [60, 40]]

print(f"最少加油次数: {min_refuel_stops(target_distance, max_fuel, stations)}") # 期望: 2
    
  
"""
第一题 (数据结构与问题建模)

主题：机场CPU任务调度

机场的一个中央处理器（CPU）一次只能处理一个任务。现有N个任务，以列表 tasks 形式给出，每个任务是 [enqueueTime, processingTime]，分别代表任务的入队时间和处理所需时间。

CPU的工作规则如下：

如果CPU空闲，且任务队列中有任务，它会选择处理时间最短的任务来执行。如果处理时间相同，则选择入队时间更早的任务。

CPU一旦开始处理一个任务，就不会中断。

任务要求： 编写一个程序，模拟CPU的处理过程，并返回所有任务被执行完毕的顺序（按任务的原始索引）。

示例: tasks = [[1,2], [2,4], [3,2], [4,1]] (原始索引 0, 1, 2, 3) 输出: [0, 2, 3, 1]

过程解释:

t=1: CPU空闲，任务0入队。CPU开始处理任务0（耗时2）。

t=2: 任务1入队。

t=3: 任务2入队。CPU完成任务0。此时队列中有任务1和2，CPU选择处理时间更短的任务2（耗时2）。

t=4: 任务3入队。

t=5: CPU完成任务2。队列中有任务1和3，CPU选择处理时间更短的任务3（耗时1）。

t=6: CPU完成任务3。队列中只剩任务1，开始处理任务1。

t=10: CPU完成任务1。所有任务结束。

执行顺序为: 0, 2, 3, 1。
"""

import heapq
def cpu_scheduler(tasks: list[list[int]]) -> list[int]:
  n = len(tasks)
  tasks_with_indices = sorted([(tasks[i][0], tasks[i][1], i) for i in range(n)])
  task_index = 0
  result_order = []
  current_time = 0
  
  available_tasks_pq = []
  
  
  while len(result_order) < n:
    while task_index < n and tasks_with_indices[task_index][1] < current_time:
      enqueueTime, processingTime, index = tasks_with_indices[task_index]
      heapq.heappush(available_tasks_pq, (processingTime, enqueueTime, index))
      task_index += 1
    
    if not available_tasks_pq and task_index < n:
      current_time = tasks_with_indices[task_index][0]
      continue
      
    if task_index < n and available_tasks_pq:
      processingTime, enqueueTime, index = heapq.heappop(available_tasks_pq)
      current_time += processingTime
      result_order.append(index)
    elif task_index == n and not available_tasks_pq:
      # 所有任务都已处理或加入队列，且队列已空
      break
  
  return result_order
