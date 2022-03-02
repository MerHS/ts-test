import torch

@torch.jit.script
def add(a, b, c, d, e, f, g, h):
    return a.add(b) \
            .add(c) \
            .add(d) \
            .add(e) \
            .add(f) \
            .add(g) \
            .add(h) 

@torch.jit.script
def add_(a, b, c, d, e, f, g, h):
    return a.add_(b) \
            .add_(c) \
            .add_(d) \
            .add_(e) \
            .add_(f) \
            .add_(g) \
            .add_(h) 

@torch.jit.script
def sub(a, b, c, d, e, f, g, h):
    return a.sub(b) \
            .sub(c) \
            .sub(d) \
            .sub(e) \
            .sub(f) \
            .sub(g) \
            .sub(h) 

@torch.jit.script
def sub_(a, b, c, d, e, f, g, h):
    return a.sub_(b) \
            .sub_(c) \
            .sub_(d) \
            .sub_(e) \
            .sub_(f) \
            .sub_(g) \
            .sub_(h) 

@torch.jit.script
def mul(a, b, c, d, e, f, g, h):
    return a.mul(b) \
            .mul(c) \
            .mul(d) \
            .mul(e) \
            .mul(f) \
            .mul(g) \
            .mul(h) 

@torch.jit.script
def mul_(a, b, c, d, e, f, g, h):
    return a.mul_(b) \
            .mul_(c) \
            .mul_(d) \
            .mul_(e) \
            .mul_(f) \
            .mul_(g) \
            .mul_(h) 

@torch.jit.script
def div(a, b, c, d, e, f, g, h):
    return a.div(b) \
            .div(c) \
            .div(d) \
            .div(e) \
            .div(f) \
            .div(g) \
            .div(h) 

@torch.jit.script
def div_(a, b, c, d, e, f, g, h):
    return a.div_(b) \
            .div_(c) \
            .div_(d) \
            .div_(e) \
            .div_(f) \
            .div_(g) \
            .div_(h) 


@torch.jit.script
def add_distl(a, b, c, d, e, f, g, h):
    return a.mul(b) + a.mul(c) + a.mul(d) + a.mul(e) + a.mul(f) + a.mul(g) + a.mul(h)


@torch.jit.script
def add_distr(a, b, c, d, e, f, g, h):
    return b.mul(a) + c.mul(a) + d.mul(a) + e.mul(a) + f.mul(a) + g.mul(a) + h.mul(a)


@torch.jit.script
def add_dist_comm(a, b, c, d, e, f, g, h, i):
    return a.mul(b) + c.mul(a) + d.mul(e) + f.mul(d) + g.mul(h) + i.mul(g)


@torch.jit.script
def x_mm(a, b, c, d, e, f, g, h):
    return a.mm(b) \
            .mm(c) \
            .mm(d) \
            .mm(e) \
            .mm(f) \
            .mm(g) \
            .mm(h) 


@torch.jit.script
def x_addmm(a, b, c, d, e, f, g, h, i):
    return a.addmm(b, c).addmm(d, e).addmm(f, g).addmm(h, i)
      
@torch.jit.script
def x_mmmul(a, b, c, d, e, f, g, h, i):
    return a.mm(b).mul(c) \
            .mm(d).mul(e) \
            .mm(f).mul(g) \
            .mm(h).mul(i)


@torch.jit.script
def x_mulmm(a, b, c, d, e, f, g, h, i):
    return a.mul(b).mm(c) \
            .mul(d).mm(e) \
            .mul(f).mm(g) \
            .mul(h).mm(i)


@torch.jit.script
def addmul(a, b, c, d, e, f, g, h, i):
    return a.add(b).mul(c) \
            .add(d).mul(e) \
            .add(f).mul(g) \
            .add(h).mul(i)

@torch.jit.script
def muladd(a, b, c, d, e, f, g, h, i):
    return a.mul(b).add(c) \
            .mul(d).add(e) \
            .mul(f).add(g) \
            .mul(h).add(i)

@torch.jit.script
def muldiv(a, b, c, d, e, f, g, h, i):
    return a.mul(b).div(c) \
            .mul(d).div(e) \
            .mul(f).div(g) \
            .mul(h).div(i)

@torch.jit.script
def divmul(a, b, c, d, e, f, g, h, i):
    return a.div(b).mul(c) \
            .div(d).mul(e) \
            .div(f).mul(g) \
            .div(h).mul(i)

@torch.jit.script
def addmul_(a, b, c, d, e, f, g, h, i):
    return a.add_(b).mul_(c) \
            .add_(d).mul_(e) \
            .add_(f).mul_(g) \
            .add_(h).mul_(i)

@torch.jit.script
def muladd_(a, b, c, d, e, f, g, h, i):
    return a.mul_(b).add_(c) \
            .mul_(d).add_(e) \
            .mul_(f).add_(g) \
            .mul_(h).add_(i)

@torch.jit.script
def muldiv_(a, b, c, d, e, f, g, h, i):
    return a.mul_(b).div_(c) \
            .mul_(d).div_(e) \
            .mul_(f).div_(g) \
            .mul_(h).div_(i)

@torch.jit.script
def divmul_(a, b, c, d, e, f, g, h, i):
    return a.div_(b).mul_(c) \
            .div_(d).mul_(e) \
            .div_(f).mul_(g) \
            .div_(h).mul_(i)


@torch.jit.script
def muldivabb(a, b, c, d, e):
    return a.mul(b).div(b) \
            .mul(c).div(c) \
            .mul(d).div(d) \
            .mul(e).div(e)

@torch.jit.script
def divmulabb(a, b, c, d, e):
    return a.div(b).mul(b) \
            .div(c).mul(c) \
            .div(d).mul(d) \
            .div(e).mul(e) 

@torch.jit.script
def addsubabb(a, b, c, d, e):
    return a.add(b).sub(b) \
            .add(c).sub(c) \
            .add(d).sub(d) \
            .add(e).sub(e) 


@torch.jit.script
def subaddabb(a, b, c, d, e):
    return a.sub(b).add(b) \
            .sub(c).add(c) \
            .sub(d).add(d) \
            .sub(e).add(e) 
