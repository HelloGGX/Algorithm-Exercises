import heapq

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


def cpu_scheduler_corrected(tasks: list[list[int]]) -> list[int]:
    n = len(tasks)
    tasks_with_indices = sorted([(tasks[i][0], tasks[i][1], i) for i in range(n)])

    current_time = 0
    result_order = []
    task_index = 0
    available_tasks_pq = []

    while len(result_order) < n:

        while task_index < n and tasks_with_indices[task_index][0] <= current_time:
            enq_time, proc_time, orig_idx = tasks_with_indices[task_index]
            heapq.heappush(available_tasks_pq, (proc_time, enq_time, orig_idx))
            task_index += 1

        if not available_tasks_pq and task_index < n:
            current_time = tasks_with_indices[task_index][0]
            continue

        if available_tasks_pq:
            proc_time, enq_time, orig_idx = heapq.heappop(available_tasks_pq)
            result_order.append(orig_idx)
            current_time += proc_time
        elif task_index == n and not available_tasks_pq:
            break

    return result_order


# 示例
tasks = [[1, 2], [2, 4], [3, 2], [4, 1]]
print(f"任务执行顺序: {cpu_scheduler_corrected(tasks)}")  # 输出: [0, 2, 3, 1]

tasks2 = [[7, 10], [7, 12], [7, 5], [7, 4], [7, 2]]
print(
    f"任务执行顺序: {cpu_scheduler_corrected(tasks2)}"
)  # 输出: [4, 3, 2, 0, 1] (按处理时间排序)
