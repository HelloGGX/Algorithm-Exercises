"""
第一题 (数据结构与算法)

主题：热门航线统计

给定一个航线列表 routes，其中每个元素是形如 "JFK-LAX" 的字符串。你的任务是找出出现频率最高的 k 条航线。

示例: routes = ["JFK-LAX", "JFK-ORD", "SFO-JFK", "JFK-LAX", "SFO-JFK", "JFK-LAX"], k = 2 输出: ["JFK-LAX", "SFO-JFK"] (顺序不重要)

提示: 这个问题考察对哈希表（字典）和堆（优先队列）的综合运用。
"""

import collections
import heapq

def top_k_frequent_routes(routes, k):
    if not routes or k == 0:
        return []
     
    # 1. 哈希表计数 (标准、直观的方式)
    freq_map = collections.Counter(routes)
    # 示例: freq_map = {'JFK-LAX': 3, 'SFO-JFK': 2, 'JFK-ORD': 1}
    print(freq_map)
    # 2. 维护一个大小为 k 的最小堆 (擂主榜)
    # 堆中存放元组 (频率, 航线)
    min_heap = []
     
    for route, freq in freq_map.items():
        if len(min_heap) < k:
            heapq.heappush(min_heap, (freq, route))
        else:
            if freq > min_heap[0][0]:
                # 新来的更强，踢掉最弱的，自己上
                heapq.heapreplace(min_heap, (freq, route))
            
    # 3. 提取结果
    # 此时堆里就是频率最高的 k 个元素
    # 我们只需要航线名，不需要频率
    result = [route for freq, route in min_heap]
    
    return result      
    
    
      
routes = ["JFK-LAX", "JFK-ORD", "SFO-JFK", "JFK-LAX", "SFO-JFK", "JFK-LAX"]  
res = top_k_frequent_routes(routes, 2)
print(res)    
             
     
     