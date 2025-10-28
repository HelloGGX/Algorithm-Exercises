"""
第二题 (链表 & 排序算法综合应用)

主题：航班旅客登机顺序重排

假设一批旅客的登机信息存储在一个双向链表中，每个节点包含旅客的 priority_level (整数，越小越优先) 和 original_order (原始在链表中的位置)。

任务要求：

提取与排序: 遍历原始的双向链表，将所有旅客信息提取出来。然后，使用一种高效的排序算法（如快速排序或归并排序）对旅客进行排序。
排序规则为：首先按 priority_level 升序排列；如果 priority_level 相同，则按 original_order 升序排列（保持稳定性）。

重构链表: 根据排序后的结果，重新构建一个新的双向链表，表示最终的登机顺序。

输出: 返回新构建的双向链表的头节点。
"""

from typing import Optional


class PassengerNode:
    def __init__(self, priority_level, original_order):
        self.priority_level = priority_level
        self.original_order = original_order
        self.prev = None
        self.next = None


# 创建双向链表mock数据
def create_mock_passenger_list():
    mock_data = [
        (2, 1),
        (1, 2),
        (3, 3),
        (1, 4),
        (2, 5),
        (0, 6),
        (3, 7),
        (2, 8),
        (1, 9),
        (0, 10),
    ]

    head = PassengerNode(mock_data[0][0], mock_data[0][1])
    current = head

    for i in range(1, len(mock_data)):
        new_node = PassengerNode(mock_data[i][0], mock_data[i][1])
        current.next = new_node
        new_node.prev = current
        current = new_node

    return head


# 直接使用
passenger_list = create_mock_passenger_list()


def sort_psg(psg_info: Optional[PassengerNode]):
    psgs = []
    while psg_info:
        psgs.append((psg_info.priority_level, psg_info.original_order))
        psg_info = psg_info.next

    print(psgs)


sort_psg(passenger_list)


"""
第三题 (查找算法 & 数组操作综合应用)

主题：寻找航班准点率阈值

你获得了一个按起飞时间排序的航班记录数组 flights，每个元素包含 [departure_time, is_on_time] (其中 is_on_time 为 1 表示准点，0 表示延误)。

任务要求： 找出第一个使得该时间点之前（不包括该时间点）的所有航班的准点率首次低于给定阈值 threshold (例如 0.8) 的航班的 departure_time。如果所有航班的准点率始终不低于阈值，则返回 -1。

示例:

flights = [[800, 1], [815, 0], [830, 1], [900, 0], [915, 0]]

threshold = 0.8

输出: 900

过程解释:

时间点 815 前: 1个航班, 1准点, 准点率 1.0 >= 0.8

时间点 830 前: 2个航班, 1准点, 准点率 0.5 < 0.8。首次低于阈值发生在830之前。触发这个变化的是时间点 830 之前的记录（到815为止）。但题目问的是导致这个变化的航班时间点。

时间点 900 前: 3个航班, 2准点, 准点率 0.667 < 0.8。

时间点 915 前: 4个航班, 2准点, 准点率 0.5 < 0.8。

修正解释: 我们需要找到第一个departure_time T，使得 [0, T) 区间内的准点率 < threshold。

T=800: 区间 [0, 800) 为空，跳过或视为满足。

T=815: 区间 [0, 815) 包含 [800, 1]。准点率 1/1 = 1.0 >= 0.8。

T=830: 区间 [0, 830) 包含 [800, 1], [815, 0]。准点率 1/2 = 0.5 < 0.8。这是第一个 T 满足条件。所以返回 830。
"""


"""
第四题 (图的遍历应用)

主题：识别关键航路节点

给定一个无向的航线网络图，表示为邻接表 graph (字典形式，key为机场编号，value为相邻机场列表)。一个“关键航路节点”（割点/Articulation Point）被定义为：如果移除该机场及其所有相连的航线，会导致整个航线网络分裂成更多的连通分量。

任务要求： 编写一个程序，找出给定航线网络中的所有关键航路节点。

示例:

graph = {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2, 4], 4: [3]}

(这是一个链状图 0-1-2-3-4，其中0,1,2构成一个三角形)

输出: [2, 3] 解释: 移除2会使0,1与3,4断开；移除3会使2与4断开。移除0,1,4不会增加连通分量数。

考纲对应: 图算法 (图的遍历：DFS、BFS实现图的连通性判断)。 提示: 这个问题通常使用基于深度优先搜索 (DFS) 的 Tarjan 算法或类似思路来解决，需要追踪节点的发现时间和最低回溯时间。
"""


"""
第五题 (图最短路径与最小生成树综合)

主题：应急航线规划

某区域发生紧急情况，需要快速建立一个临时的应急航线网络，并确定两个关键城市间的通信延迟。给定该区域所有 N 个城市以及一份备选的双向航线列表 potential_routes，每条记录为 [城市A, 城市B, 通信延迟] (延迟即成本/权重)。

任务要求：

构建应急网络 (MST): 首先，确定一个能连接所有城市，且总通信延迟最低的应急网络方案（即构建最小生成树）。

关键路径分析 (Shortest Path on MST): 在你上一步构建好的应急网络 (MST) 上，计算从指定的 城市S 到 城市D 的通信延迟最低的路径。输出这条路径的总延迟。

示例:

N=5

potential_routes = [[0,1,1], [0,2,7], [1,2,5], [1,3,4], [1,4,3], [2,4,6], [3,4,2]]

S=0, D=4

输出:

(MST构建过程，假设使用Kruskal) 选边: [0,1,1], [3,4,2], [1,4,3], [1,3,4] (舍弃)。 MST总延迟 = 1+2+3 = 6。 MST包含边: (0,1), (3,4), (1,4)。

在MST上从0到4的最短路径: 0 -> 1 -> 4。总延迟: 1 + 3 = 4。
"""
