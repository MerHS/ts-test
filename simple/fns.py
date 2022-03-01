def add(a, b):
    return a.add(b)

def add_(a, b):
    return a.add_(b)

def sub(a, b):
    return a.sub(b)

def sub_(a, b):
    return a.sub_(b)

def mul(a, b):
    return a.mul(b)

def mul_(a, b):
    return a.mul_(b)

def div(a, b):
    return a.div(b)

def div_(a, b):
    return a.div_(b)

def mm(a, b):
    return a.mm(b)

def mm_(a, b):
    return a.mm_(b)

def addmul(a, b, c):
    return a.add(b).mul(c)

def muladd(a, b, c):
    return a.mul(b).add(c)

def muldiv(a, b, c):
    return a.mul(b).div(c)

def divmul(a, b, c):
    return a.div(b).mul(c)

def addmul_(a, b, c):
    return a.add_(b).mul_(c)

def muladd_(a, b, c):
    return a.mul_(b).add_(c)

def muldiv_(a, b, c):
    return a.mul_(b).div_(c)

def divmul_(a, b, c):
    return a.div_(b).mul_(c)

def muldivaba(a, b):
    return a.mul(b).div(a)

def muldivabb(a, b):
    return a.mul(b).div(b)

def divmulabb(a, b):
    return a.div(b).mul(b)

def addsubabb(a, b):
    return a.add(b).sub(b)

def addsubaba(a, b):
    return a.add(b).sub(a)

def subaddabb(a, b):
    return a.sub(b).add(b)

def muldivaba_(a, b):
    return a.mul_(b).div_(a)

def muldivabb_(a, b):
    return a.mul_(b).div_(b)

def divmulabb_(a, b):
    return a.div_(b).mul_(b)

def addsubabb_(a, b):
    return a.add_(b).sub_(b)

def addsubaba_(a, b):
    return a.add_(b).sub_(a)

def subaddabb_(a, b):
    return a.sub_(b).add_(b)