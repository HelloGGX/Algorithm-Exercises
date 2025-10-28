"""
● 输入指令: "MAX(3 * 4, 5 + MAX(2, 8))"
● 后缀表达式 (一种可能): 3 4 * 5 2 8 MAX + MAX
● 最终优先级分数: 13 (计算过程: 3*4=12; MAX(2,8)=8; 5+8=13; MAX(12, 13)=13)
"""
def tokenize(expr:str):
    tokens = []
    i = 0
    
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        elif ch.isdigit():
            num = ch
            i += 1
            while i < len(expr) and expr[i].isdigit():
                num += expr[i]
                i += 1
            tokens.append(num)
        elif ch.isalpha():
            func = ch
            i += 1
            while i < len(expr) and expr[i].isalpha():
                func += expr[i]
                i += 1
            tokens.append(func)
        else:
            tokens.append(ch)
            i += 1
    
    print(tokens)
    return tokens

def shunting_yard(expr):
    if not expr:
        return ''
    prec = {'+': 1, '-': 1, '*': 2, '/': 2}
    output = []
    stack = []
    arg_count = []  # 参数计数栈
    # ['MAX', '(', '3', '*', '4', ',', '5', '+', 'MAX', '(', '2', ',', '8', ')', ')']
    tokens = tokenize(expr)
    
    for token in tokens:
        if token.isdigit():
            output.append(token)
        elif token.isalpha():
            stack.append(token)
        elif token == '(':
            # 如果前一个是函数名，则初始化参数计数器
            if stack and stack[-1].isalpha():
                arg_count.append(1)
            stack.append(token)
        elif token == ',':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if arg_count:
                arg_count[-1] += 1
        elif token in prec:
            while stack and stack[-1] in prec and prec[stack[-1]] >= prec[token]:
                output.append(stack.pop())
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
            # 如果 '(' 之前有函数名，则弹出函数名
            if stack and stack[-1].isalpha():
                func = stack.pop()
                n_args = arg_count.pop() if arg_count else 0
                output.append((func, n_args))
        
    while stack:
        output.append(stack.pop())
    
    print(output)
    return output

# ['3', '4', '*', '5', '2', '8', '9', ('MAX', 3), '+', ('MAX', 2)]
def eval_postfix(tokens):
    stack = []
    for token in tokens:
        if token.isdigit():
            stack.append(int(token))
        elif token in ('+', '-', '*', '/'):
            b = stack.pop()
            a = stack.pop()
            if token == '+': stack.append(a+b)
            elif token == '*': stack.append(a*b)
            elif token == '-': stack.append(a-b)
            elif token == '/': stack.append(a/b)
        elif isinstance(token, tuple):
            func, n_args = token
            args = [stack.pop() for _ in range(n_args)][::1]
            if func == 'MAX':
                stack.append(max(args))
            elif func == 'MIN':
                stack.append(min(args))
            else:
                raise ValueError(f"未知函数: {func}")
        else:
            raise ValueError(f"未知 token: {token}")
        
    if len(stack) != 1:
        raise ValueError("表达式错误，栈未清空")
    return stack[0]
