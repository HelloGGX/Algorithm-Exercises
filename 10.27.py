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

"""


def gate_scheduler(G: int, flights: list[list[int]]) -> int:
    if not flights:
        return 0

    flights.sort(key=lambda x: x[0])

    # [(最早完成时间， 登机口编号)]
    gate_heap = [(0, i) for i in range(G)]
    heapq.heapify(gate_heap)

    max_finish_time = 0

    for req_time, duration in flights:
        gate_finish_time, gate_id = heapq.heappop(gate_heap)
        start_time = max(req_time, gate_finish_time)
        finish_time = start_time + duration
        max_finish_time = max(max_finish_time, finish_time)
        heapq.heappush(gate_heap, (finish_time, gate_id))

    return max_finish_time


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
    if not tasks:
        return []

    total_task = len(tasks)
    tasks_with_indices = sorted(
        [(tasks[i][0], tasks[i][1], i) for i in range(total_task)]
    )

    current_time = 0
    task_idx = 0
    result_order = []
    task_heap = []

    while len(result_order) < total_task:
        while task_idx < total_task and tasks_with_indices[task_idx][0] <= current_time:
            enqueueTime, processingTime, orig_idx = tasks_with_indices[task_idx]
            heapq.heappush(task_heap, (processingTime, enqueueTime, orig_idx))
            task_idx += 1

        if not task_heap and task_idx < total_task:
            current_time = tasks_with_indices[task_idx][0]
            continue

        if task_heap:
            processingTime, enqueueTime, orig_idx = heapq.heappop(task_heap)
            result_order.append(orig_idx)
            current_time += processingTime
        elif task_idx == total_task and not task_heap:
            break

    return result_order


"""W
第二题 (传统算法与问题建模)

主题：寻找最佳加油时机

一架飞机需要执行一段长途飞行任务，起点油箱是满的，可以飞行 start_fuel 公里。沿途有多个加油站，信息存储在列表 stations 中，每个元素是 [distance_from_start, fuel_capacity]。

你的任务是：计算飞机从起点到达终点 target_distance 所需的最少加油次数。如果无法到达，则返回 -1。你可以假设在任何加油站加油都会把油箱加满。

示例:

target_distance = 100, start_fuel = 10

stations = [[10, 60], [20, 30], [30, 30], [60, 40]]

输出: 2

过程解释:

从起点出发，油箱能飞10公里，到达加油站[10, 60]。

在这里加油（第1次），油箱加满，能飞到 10 + 60 = 70 公里处。

在能到达的范围内（70公里内），选择下一个能让你飞得最远的加油站。在[20,30], [30,30], [60,40]中，显然在[60,40]加油最划算。

飞到[60,40]加油（第2次），油箱加满，能飞到 60 + 40 = 100 公里处，正好到达终点。
"""


def min_stops(target_distance: int, start_fuel: int, stations: list[list[int]]) -> int:
    if target_distance <= start_fuel:
        return 0

    stations.sort(key=lambda x: x[1])
    print(stations)
    current_distance = start_fuel
    station_heap = []
    stops = 0

    while current_distance < target_distance:
        # 找出能飞到的加油站的距离
        for distance_from_start, fuel_capacity in stations:
            if current_distance >= distance_from_start:
                heapq.heappush(station_heap, -fuel_capacity)

        if station_heap:
            fuel_capacity = -heapq.heappop(station_heap)
            current_distance += fuel_capacity

        stops += 1

    print(stops)
    return stops


min_stops(100, 10, [[10, 60], [20, 30], [30, 30], [60, 40]])


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
from collections import defaultdict


def assign_luggage(N: int, baggage) -> int:
    if not baggage:
        return 0

    waiting_bags = defaultdict(list)
    for start, target in baggage:
        waiting_bags[start].append(target)

    # [(current_pos, target_pos)]
    on_belt = []
    total_bags = len(baggage)
    delivered_count = 0
    time = 0

    while delivered_count < total_bags:
        # 移动
        for i in range(len(on_belt)):
            current_pos, target_pos = on_belt[i]
            on_belt[i] = ((current_pos + 1) % N, target_pos)

        # 下行李
        remained_bags = []
        for current_pos, target_pos in on_belt:
            if current_pos == target_pos:
                delivered_count += 1
            else:
                remained_bags.append((current_pos, target_pos))

        on_belt = remained_bags

        # 上行李
        current_station_pos = time % N
        if current_station_pos in waiting_bags:
            for target in waiting_bags[current_station_pos]:
                on_belt.append((current_station_pos, target))

            del waiting_bags[current_station_pos]

        time += 1

    print(time)
    return time


baggage = [(0, 2), (1, 3)]
assign_luggage(5, baggage)
