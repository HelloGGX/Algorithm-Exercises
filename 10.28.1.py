"""
第一题 (栈 & 树结构综合应用)

主题：航空指令优先级计算器
你需要实现一个计算器，用于评估一系列嵌套航空指令的最终优先级分数。指令以特殊的中缀表达式字符串给出，包含数字（代表基础优先级）、操作符（+ 加法, * 乘法）以及括号。特别地，该计算器引入了一个最大值函数 MAX(arg1, arg2, ...)，可以接受任意数量的参数。

任务要求：
解析与转换: 使用栈的原理，将给定的中缀表达式字符串转换为后缀表达式（逆波兰表示法），需要正确处理 MAX 函数及其不定数量的参数，以及 +, * 的运算优先级（乘法优先于加法）。
后缀表达式求值: 基于生成的后缀表达式，使用栈进行求值，计算出最终的优先级分数。

示例:

输入指令: "MAX(3 * 4, 5 + MAX(2, 8))"
后缀表达式 (一种可能): 3 4 * 5 2 8 MAX + MAX

最终优先级分数: 13 (计算过程: 3*4=12; MAX(2,8)=8; 5+8=13; MAX(12, 13)=13)
● 输入指令: "MAX(3 * 4, 5 + MAX(2, 8))"
● 后缀表达式 (一种可能): 3 4 * 5 2 8 MAX + MAX
● 最终优先级分数: 13 (计算过程: 3*4=12; MAX(2,8)=8; 5+8=13; MAX(12, 13)=13)
"""


def tokenize(expr: str):
    stack = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            num = ch
            i += 1
            while i < len(expr) and expr[i].isdigit():
                num += expr[i]
                i += 1
            stack.append(num)
        elif ch.isalpha():
            func = ch
            i += 1
            while i < len(expr) and expr[i].isalpha():
                func += expr[i]
                i += 1
            stack.append(func)
        else:
            stack.append(ch)
            i += 1

    return stack


def shunting_yard(expr):
    if not expr:
        return []

    output = []
    stack = []
    prec = {"+": 1, "-": 1, "*": 2, "/": 2}
    arg_count = []  # 参数计数栈
    # ['MAX', '(', '3', '*', '4', ',', '5', '+', 'MAX', '(', '2', ',', '8', ')', ')']
    tokens = tokenize(expr)

    for token in tokens:
        if token.isdigit():
            output.append(token)
        elif token.isalpha():
            stack.append(token)
        elif token == "(":
            if stack and stack[-1].isalpha():
                arg_count.append(1)
            stack.append(token)
        elif token == ",":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if arg_count:
                arg_count[-1] += 1
        elif token in prec:
            while stack and stack[-1] in prec and prec[stack[-1]] >= prec[token]:
                output.append(stack.pop())
            stack.append(token)
        elif token == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            stack.pop()
            if stack and stack[-1].isalpha():
                func = stack.pop()
                n_args = arg_count.pop() if arg_count else 0
                output.append((func, n_args))

    while stack:
        output.append(stack.pop())

    return output


def eval_postfix(tokens):
    stack = []
    for token in tokens:
        if isinstance(token, str) and token.isdigit():
            stack.append(int(token))
        elif isinstance(token, str) and token in ("+", "-", "*", "/"):
            b = stack.pop()
            a = stack.pop()
            if token == "+":
                stack.append(a + b)
            elif token == "*":
                stack.append(a * b)
            elif token == "-":
                stack.append(a - b)
            elif token == "/":
                stack.append(a / b)
        elif isinstance(token, tuple):
            func, n_args = token
            args = [stack.pop() for _ in range(n_args)][::-1]
            if func == "MAX":
                stack.append(max(args))
            elif func == "MIN":
                stack.append(min(args))
            else:
                raise ValueError(f"未知函数: {func}")
        else:
            raise ValueError(f"未知 token: {token}")

    if len(stack) != 1:
        raise ValueError("表达式错误，栈未清空")
    return stack[0]


tokens = shunting_yard("MAX(3 * 4, 5 + MAX(2, 8))")
print(tokens)
res = eval_postfix(tokens)
print(res)
