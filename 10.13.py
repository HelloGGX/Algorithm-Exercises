"""
第一题 (基础数据结构 - 栈)

主题：航班高度监控

给定一个表示航班在不同时间点高度的数组 altitudes。
对于每个时间点的高度，找出其右侧第一个比它更高的高度。如果不存在，则结果为-1。

示例: altitudes = [73, 74, 75, 71, 69, 72, 76]
输出: [74, 75, 76, 72, 72, 76, -1]
提示: 这是单调栈的另一个经典应用。
"""

def next_higher_altitudes(altitudes):
    if not altitudes:
        return []
    stack = []
    res = [-1] * len(altitudes) #[74, 75, 75, 72, 72, 76, -1]
    for i,  cur_altitude in enumerate(altitudes):
        
        while stack and stack[-1][1] <= cur_altitude:
             top_stack = stack.pop()
             top_stack_index = top_stack[0]
             res[top_stack_index] = cur_altitude
        
            
        stack.append((i, cur_altitude))
    return res

res = next_higher_altitudes([73, 74, 75, 71, 69, 72, 76])
print(res)
    
    
"""
第二题 (树与递归)

主题：航线网络合法性校验
航线网络规划中，有时会将航点组织成二叉树结构以便快速查找。
一个合法的“航点查找树”必须是一个有效的二叉搜索树 (BST)。
你的任务是：编写一个函数，判断给定的一个二叉树是否为有效的二叉搜索树。

BST定义:

节点的左子树只包含小于当前节点的数。
节点的右子树只包含大于当前节点的数。
所有左、右子树也必须是二叉搜索树。
提示: 需要在递归中传递上界和下界信息，考察对递归状态的深入理解。
"""
from typing import Optional

# 我的错误解法
# class Node:
#     def __init__(self, val):
#         self.left: Optional[Node]  = None
#         self.right: Optional[Node] = None
#         self.val = val

# def search_tree(root: Optional[Node]):
#     # 如果达到叶子节点
#     if root and not root.left and not root.right:
#         return root
    
#     left_node = search_tree(root.left)
#     right_node = search_tree(root.right)
    
#     if left_node.val < root.val < right_node.val:
#         return True
#     else:
#         return False

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
         return self._is_valid_helper(root, float('-inf'), float('inf'))
    
    def _is_valid_helper(self, node: Optional[TreeNode], lower:float, upper: float) -> bool:
        if not node:
            return True
        
        if not (lower < node.val < upper):
            return False
        
        return (self._is_valid_helper(node.left, lower, node.val)) and (self._is_valid_helper(node.right, node.val, upper))