import torch

@torch.jit.script
def add(a, b):
    return a.add(b)

@torch.jit.script
def add_(a, b):
    return a.add_(b)

@torch.jit.script
def sub(a, b):
    return a.sub(b)

@torch.jit.script
def sub_(a, b):
    return a.sub_(b)

@torch.jit.script
def mul(a, b):
    return a.mul(b)

@torch.jit.script
def mul_(a, b):
    return a.mul_(b)

@torch.jit.script
def div(a, b):
    return a.div(b)

@torch.jit.script
def div_(a, b):
    return a.div_(b)

@torch.jit.script
def mm(a, b):
    return a.mm(b)

@torch.jit.script
def mm_(a, b):
    return a.mm_(b)

@torch.jit.script
def addmul(a, b, c):
    return a.add(b).mul(c)

@torch.jit.script
def muladd(a, b, c):
    return a.mul(b).add(c)

@torch.jit.script
def muldiv(a, b, c):
    return a.mul(b).div(c)

@torch.jit.script
def divmul(a, b, c):
    return a.div(b).mul(c)

@torch.jit.script
def addmul_(a, b, c):
    return a.add_(b).mul_(c)

@torch.jit.script
def muladd_(a, b, c):
    return a.mul_(b).add_(c)

@torch.jit.script
def muldiv_(a, b, c):
    return a.mul_(b).div_(c)

@torch.jit.script
def divmul_(a, b, c):
    return a.div_(b).mul_(c)

@torch.jit.script
def muldivaba(a,b):
    return a.mul(b).div(a)

@torch.jit.script
def muldivabb(a, b):
    return a.mul(b).div(b)

@torch.jit.script
def divmulabb(a, b):
    return a.div(b).mul(b)

@torch.jit.script
def addsubabb(a, b):
    return a.add(b).sub(b)

@torch.jit.script
def addsubaba(a, b):
    return a.add(b).sub(a)

@torch.jit.script
def subaddabb(a, b):
    return a.sub(b).add(b)

@torch.jit.script
def muldivaba_(a, b):
    return a.mul_(b).div_(a)

@torch.jit.script
def muldivabb_(a, b):
    return a.mul_(b).div_(b)

@torch.jit.script
def divmulabb_(a, b):
    return a.div_(b).mul_(b)

@torch.jit.script
def addsubabb_(a, b):
    return a.add_(b).sub_(b)

@torch.jit.script
def addsubaba_(a, b):
    return a.add_(b).sub_(a)

@torch.jit.script
def subaddabb_(a, b):
    return a.sub_(b).add_(b)
    