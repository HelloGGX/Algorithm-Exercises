"""
第一题 (数据结构 - 30% 分值区)

主题：航班调度指令解析器

你需要编写一个程序，用于解析和执行一系列嵌套的航班调度指令。指令以字符串形式给出，格式类似于数学表达式，包含指令（单个大写字母，代表具体操作）、参数（数字）和括号，用于控制执行优先级。

任务要求：

指令解析: 使用栈的原理，将给定的中缀表达式形式的指令字符串（例如 A(1,B(2,3))）转换为后缀表达式（逆波兰表示法）。

构造执行树: 根据后缀表达式，构建一棵二叉表达式树。在这棵树中，叶子节点是参数（数字），非叶子节点是指令（操作符）。

执行指令: 对构建好的二叉树进行后序遍历，模拟指令的执行并返回最终结果。假设指令 A(x,y) 执行 x+y，指令 B(x,y) 执行 x*y。

示例:

输入指令: "A(5,B(3,4))"

后缀表达式应为: 5 3 4 B A

最终执行结果: 17 (计算过程为 3 * 4 = 12, 然后 5 + 12 = 17)

考纲对应:

栈与队列应用: 基于栈实现表达式求值。

树结构操作: 构造简单的二叉树结构，完成...后序遍历。
"""

def infix_to_postfix(commads:str):
    # A(5,B(3,4))
    operators = {'A', 'B'}
    op_stack = []
    postfix_list = []
    
    for commad in commads:
        if commad in operators:
            op_stack.append(commad)
        elif commad.isdigit():
            postfix_list.append(commad)
        elif commad == ')':
            if op_stack:
                postfix_list.append(op_stack.pop())
    return postfix_list

infix_to_postfix('A(5,B(3,4))')


def evaluate_postfix(postfix_list: list):
    stack = []
    for item in postfix_list:
        if item.isdigit():
            stack.append(int(item))
        elif item == 'B':
            y = stack.pop()
            x = stack.pop()
            stack.append(x * y)
        elif item == 'A':
            y = stack.pop()
            x = stack.pop()
            stack.append(x + y)
            
    return stack[0]


res = evaluate_postfix(infix_to_postfix('A(5,B(3,4))'))
print(res)