import heapq


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
