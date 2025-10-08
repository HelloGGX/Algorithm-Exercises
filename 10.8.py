"""
题目描述：实现一个双向链表结构，用于模拟民航航班链（每个节点包含航班号、起飞时间、目的地）。输入一个航班列表（e.g., [[101, '08:00', '北京'], [202, '09:00', '上海'], ...]），构建链表；然后实现插入新航班（按起飞时间顺序）、删除指定航班号节点，并输出最终链表遍历结果（从头到尾）。额外要求：计算链表长度并分析时间复杂度。
极大可能的考察方向：链表结构实现，包括遍历、插入与删除（大纲中明确提及），结合民航场景模拟资源管理。
需要着重练习的地方：链表节点的类定义（prev/next指针）、边界情况处理（如空链表、头尾插入），以及时间复杂度说明。练习使用Python类实现，避免数组混淆；多测试插入/删除后的完整性，以防指针错误（常见扣分点）。
"""
from typing import Optional
from datetime import datetime
import time

class Node:
    def __init__(self, flight_number: int, departure_time: str, destination: str):
        self.flight_number = flight_number
        self.departure_time = departure_time  # e.g., '08:00'
        self.destination = destination
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

class DoublyLinkedList:
    def __init__(self):
        """初始化双向链表，使用哨兵节点简化边界处理"""
        self.head = Node(0, '00:00', '')
        self.tail = Node(0, '23:59', '')
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
        # 哈希表存储flight_number到节点的映射，便于快速删除
        self.flight_map = {}

    def _parse_time(self, time_str: str) -> int:
        """将时间字符串（如'08:00'）转换为分钟数，便于比较"""
        return int(datetime.strptime(time_str, '%H:%M').hour * 60 + datetime.strptime(time_str, '%H:%M').minute)

    def insert(self, flight_number: int, departure_time: str, destination: str):
        """插入新航班，按起飞时间顺序"""
        start_time = time.perf_counter()
        new_node = Node(flight_number, departure_time, destination)
        time_minutes = self._parse_time(departure_time)

        # 从头遍历找到插入位置
        current = self.head.next
        while current != self.tail and self._parse_time(current.departure_time) < time_minutes:
            current = current.next
        
        # 插入节点
        new_node.next = current
        new_node.prev = current.prev
        current.prev.next = new_node
        current.prev = new_node
        self.flight_map[flight_number] = new_node
        self.size += 1

        end_time = time.perf_counter()
        print(f"Insert time: {end_time - start_time:.6f}s")

    def delete(self, flight_number: int):
        """删除指定航班号的节点"""
        start_time = time.perf_counter()
        if flight_number not in self.flight_map:
            print(f"Flight {flight_number} not found")
            return
        
        node = self.flight_map[flight_number]
        node.prev.next = node.next
        node.next.prev = node.prev
        del self.flight_map[flight_number]
        self.size -= 1

        end_time = time.perf_counter()
        print(f"Delete time: {end_time - start_time:.6f}s")

    def print_list(self):
        """打印链表，从头到尾"""
        current = self.head.next
        result = []
        while current != self.tail:
            result.append([current.flight_number, current.departure_time, current.destination])
            current = current.next
        print(f"Flight list: {result}")

    def get_size(self) -> int:
        """返回链表长度"""
        return self.size

    def analyze_complexity(self):
        """时间复杂度分析"""
        print("Time Complexity Analysis:")
        print("- Insert: O(n) for traversing to find position, O(1) for insertion")
        print("- Delete: O(1) with hash map lookup, O(1) for deletion")
        print("- Print: O(n) for traversal")
        print("- Get size: O(1)")

# 测试代码
if __name__ == "__main__":
    dll = DoublyLinkedList()
    flights = [[101, '08:00', '北京'], [202, '09:00', '上海'], [303, '07:30', '广州']]
    for flight in flights:
        dll.insert(flight[0], flight[1], flight[2])
    dll.print_list()  # 应按时间排序: [303, '07:30', '广州'], [101, '08:00', '北京'], [202, '09:00', '上海']
    dll.delete(101)
    dll.print_list()
    print(f"List size: {dll.get_size()}")
    dll.analyze_complexity()
            
    
"""
题目2: 快速排序实现与性能对比（传统算法，30分）
题目描述：实现快速排序算法，对一个无序整数数组（e.g., 模拟民航旅客ID列表）进行排序。输入数组大小N（1≤N≤1000）和数组元素；输出每轮分区后的中间结果、最终排序数组，以及与插入排序的时间复杂度对比（用简单计时函数测量运行时间）。要求处理重复元素并确保稳定性。
极大可能的考察方向：排序算法实现，包括快速排序和复杂度分析（大纲中指定冒泡/选择/插入/快速/归并），强调输出每轮结果和时间对比。
需要着重练习的地方：分区函数（pivot选择，避免最坏情况如已排序数组）、递归实现和栈溢出防范。练习用time模块测量运行时间，并比较不同输入规模的复杂度；重点优化pivot为随机或中位数，以提升稳定性（高分关键）。
"""       

import time

def partition(arr, low, high):
    """分区函数，选择三数取中pivot"""
    # 三数取中：比较low、mid、high，选择中间值
    mid = (low + high) // 2
    pivot_candidates = [(arr[low], low), (arr[mid], mid), (arr[high], high)]
    pivot_candidates.sort()  # 按值排序
    pivot_idx = pivot_candidates[1][1]  # 取中间值的索引
    pivot = arr[pivot_idx]
    
    # 将pivot放到high位置
    arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
    
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:  # <= 确保重复元素靠左，增强稳定性
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quick_sort(arr, low, high):
    """快速排序，输出每轮分区结果"""
    if low < high:
        p = partition(arr, low, high)
        print(f"Partition result: {arr[:p]} | {arr[p]} | {arr[p+1:]}")
        quick_sort(arr, low, p - 1)
        quick_sort(arr, p + 1, high)
    return arr

def insertion_sort(arr):
    """插入排序，用于时间对比"""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def compare_sorting(arr):
    """比较快速排序和插入排序的运行时间"""
    # 复制数组，避免修改原数组
    arr_quick = arr.copy()
    arr_insert = arr.copy()
    
    # 快速排序
    start_time = time.perf_counter()
    quick_sort(arr_quick, 0, len(arr_quick) - 1)
    quick_time = time.perf_counter() - start_time
    
    # 插入排序
    start_time = time.perf_counter()
    insertion_sort(arr_insert)
    insert_time = time.perf_counter() - start_time
    
    print(f"Quick Sort time: {quick_time:.6f}s")
    print(f"Insertion Sort time: {insert_time:.6f}s")
    print(f"Final sorted array: {arr_quick}")
    print("Time Complexity Analysis:")
    print("- Quick Sort: Average O(n log n), Worst O(n²)")
    print("- Insertion Sort: O(n²)")

# 测试代码
if __name__ == "__main__":
    arr = [303, 101, 202, 101, 404]  # 模拟旅客ID，包含重复元素
    print(f"Original array: {arr}")
    compare_sorting(arr)
    

"""
练习题目1: 航班调度栈模拟（数据结构，30分）
题目描述：实现一个栈结构，模拟民航调度中心处理紧急航班任务。输入任务列表（e.g., [[101, '08:00', '紧急'], [202, '09:00', '普通'], ...]），按优先级（紧急>普通）和时间顺序入栈。实现push（入栈）、pop（出栈）、peek（查看栈顶），输出处理顺序（从栈顶到栈底）。计算栈操作的时间复杂度。
考察方向：栈应用（大纲指定栈实现表达式求值/任务调度），结合民航调度场景。
"""

from heapq import heappush, heappop

class FlightStack:
    def __int__(self):
        self.stack = []
    
    def push(self, flight_number: int, departure_time: str, priority: str):
        """按优先级（紧急>普通）和时间顺序入栈"""
        # 优先级：紧急=-1，普通=0
        heappush(self.stack, (-(priority == '紧急'), departure_time, flight_number))
    
    def pop(self):
        """出栈"""
        if self.stack:
            _, time, flight = heappop(self.stack)
            return flight, time
        return None
    
    def peek(self):
        if self.stack:
            _, time, flight = self.stack[0]
            return flight, time
        
        
"""
练习题目2: 航线树遍历（数据结构，30分）
题目描述：实现二叉树结构，模拟民航航线网络（节点为城市，边为航线）。输入城市对列表（e.g., [['北京', '上海'], ['上海', '广州']]），构建二叉树，实现前序和中序遍历，输出从‘北京’到‘广州’的路径。计算遍历复杂度。
考察方向：树结构操作（大纲指定二叉树遍历/路径查找），模拟航线层级。
练习重点:

实现TreeNode类和递归遍历（前序/中序）。
练习路径查找（DFS或BFS）。
处理空树/单一节点。
分析复杂度（遍历O(n)，路径查找O(n)）。
"""
class TreeNode:
    def __init__(self, city):
        self.city = city
        self.left = None
        self.right = None

class FlightTree:
    def preorder(self, root):
        if root:
            print("Preorder traversal:", root.city)
            self.preorder(root.left)
            self.preorder(root.right)
    
    def find_path(self, root, target):
        if not root:
            return []
        if root.city == target:
            return [target]
        
        left_path = self.find_path(root.left, target)
        if left_path:
            return [root.city] + left_path
        right_path = self.find_path(root.right, target)
        if right_path:
            return [root.city] + right_path
        return []
        
        
"""
练习题目3: 二分查找优化航班查询（传统算法，30分）
题目描述：实现二分查找，查询民航航班ID列表（已排序，e.g., [101, 202, 303, ...]）中的目标ID。输入N（1≤N≤1000）和查询次数K（1≤K≤100），输出每个查询的索引或-1（未找到）。优化查找第一个/最后一个重复ID，比较与顺序查找的时间。输出复杂度分析。
考察方向：查找算法实现（大纲指定二分查找/区间查找），模拟航班ID查询。
练习重点:

实现标准二分查找和变种（找第一个/最后一个）。
用time.perf_counter()比较二分与顺序查找。
处理重复ID（返回边界索引）。
分析复杂度（二分O(log n)，顺序O(n)）。
"""
def binary_search_first(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result


"""
练习题目4: 旅客满意度逻辑回归（机器学习，20分）
题目描述：给定旅客数据集（CSV，特征：年龄、航班延误，标签：满意/不满意），实现数据预处理（缺失值均值填充、标准化），训练逻辑回归模型，预测测试集满意度。输出准确率、ROC曲线（AUC值）。比较不同学习率的效果。
考察方向：数据预处理与模型训练（大纲指定逻辑回归/评估指标），模拟旅客分类。
练习重点:

用pandas/sklearn处理数据（SimpleImputer, StandardScaler）。
训练LogisticRegression，计算roc_auc_score。
调参（学习率C=0.1, 1, 10）。
练习混淆矩阵和ROC绘制。
"""

