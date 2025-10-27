"""
● 输入指令: "MAX(3 * 4, 5 + MAX(2, 8))"
● 后缀表达式 (一种可能): 3 4 * 5 2 8 MAX + MAX
● 最终优先级分数: 13 (计算过程: 3*4=12; MAX(2,8)=8; 5+8=13; MAX(12, 13)=13)
"""

import re


def command_priority(command: str) -> int:

    def handle_max(command):
        pattern = r"MAX\((\d+), (\d+)\)"  # 修正：转义括号，简化分组
        match = re.search(pattern, command)
        if match:
            return max(int(match.group(1)), int(match.group(2)))

    # [3,4,5,2,8]
    number_task = []
    # 这里放['MAX', '*', '+', 'MAX']
    commad_task = []

    for str in command:
        if str == "M":
            commad_task.append("MAX")
        elif str.isdigit():
            number_task.append(int(str))
        elif str == "*" or str == "+":
            commad_task.append(str)


command_priority("MAX(3 * 4, 5 + MAX(2, 8))")
