def add(a, b, c, d, e, f, g, h):
    return a.add(b) \
            .add(c) \
            .add(d) \
            .add(e) \
            .add(f) \
            .add(g) \
            .add(h) 


def add_(a, b, c, d, e, f, g, h):
    return a.add_(b) \
            .add_(c) \
            .add_(d) \
            .add_(e) \
            .add_(f) \
            .add_(g) \
            .add_(h) 

def sub(a, b, c, d, e, f, g, h):
    return a.sub(b) \
            .sub(c) \
            .sub(d) \
            .sub(e) \
            .sub(f) \
            .sub(g) \
            .sub(h) 

def sub_(a, b, c, d, e, f, g, h):
    return a.sub_(b) \
            .sub_(c) \
            .sub_(d) \
            .sub_(e) \
            .sub_(f) \
            .sub_(g) \
            .sub_(h) 

def mul(a, b, c, d, e, f, g, h):
    return a.mul(b) \
            .mul(c) \
            .mul(d) \
            .mul(e) \
            .mul(f) \
            .mul(g) \
            .mul(h) 

def mul_(a, b, c, d, e, f, g, h):
    return a.mul_(b) \
            .mul_(c) \
            .mul_(d) \
            .mul_(e) \
            .mul_(f) \
            .mul_(g) \
            .mul_(h) 

def div(a, b, c, d, e, f, g, h):
    return a.div(b) \
            .div(c) \
            .div(d) \
            .div(e) \
            .div(f) \
            .div(g) \
            .div(h) 

def div_(a, b, c, d, e, f, g, h):
    return a.div_(b) \
            .div_(c) \
            .div_(d) \
            .div_(e) \
            .div_(f) \
            .div_(g) \
            .div_(h) 


def add_distl(a, b, c, d, e, f, g, h):
    return a.mul(b) + a.mul(c) + a.mul(d) + a.mul(e) + a.mul(f) + a.mul(g) + a.mul(h)


def add_distr(a, b, c, d, e, f, g, h):
    return b.mul(a) + c.mul(a) + d.mul(a) + e.mul(a) + f.mul(a) + g.mul(a) + h.mul(a)


def add_dist_comm(a, b, c, d, e, f, g, h, i):
    return a.mul(b) + c.mul(a) + d.mul(e) + f.mul(d) + g.mul(h) + i.mul(g)


def x_mm(a, b, c, d, e, f, g, h):
    return a.mm(b) \
            .mm(c) \
            .mm(d) \
            .mm(e) \
            .mm(f) \
            .mm(g) \
            .mm(h) 


def x_addmm(a, b, c, d, e, f, g, h, i):
    return a.addmm(b, c).addmm(d, e).addmm(f, g).addmm(h, i)
      
def x_mmmul(a, b, c, d, e, f, g, h, i):
    return a.mm(b).mul(c) \
            .mm(d).mul(e) \
            .mm(f).mul(g) \
            .mm(h).mul(i)


def x_mulmm(a, b, c, d, e, f, g, h, i):
    return a.mul(b).mm(c) \
            .mul(d).mm(e) \
            .mul(f).mm(g) \
            .mul(h).mm(i)


def addmul(a, b, c, d, e, f, g, h, i):
    return a.add(b).mul(c) \
            .add(d).mul(e) \
            .add(f).mul(g) \
            .add(h).mul(i)

def muladd(a, b, c, d, e, f, g, h, i):
    return a.mul(b).add(c) \
            .mul(d).add(e) \
            .mul(f).add(g) \
            .mul(h).add(i)

def muldiv(a, b, c, d, e, f, g, h, i):
    return a.mul(b).div(c) \
            .mul(d).div(e) \
            .mul(f).div(g) \
            .mul(h).div(i)

def divmul(a, b, c, d, e, f, g, h, i):
    return a.div(b).mul(c) \
            .div(d).mul(e) \
            .div(f).mul(g) \
            .div(h).mul(i)

def addmul_(a, b, c, d, e, f, g, h, i):
    return a.add_(b).mul_(c) \
            .add_(d).mul_(e) \
            .add_(f).mul_(g) \
            .add_(h).mul_(i)

def muladd_(a, b, c, d, e, f, g, h, i):
    return a.mul_(b).add_(c) \
            .mul_(d).add_(e) \
            .mul_(f).add_(g) \
            .mul_(h).add_(i)

def muldiv_(a, b, c, d, e, f, g, h, i):
    return a.mul_(b).div_(c) \
            .mul_(d).div_(e) \
            .mul_(f).div_(g) \
            .mul_(h).div_(i)

def divmul_(a, b, c, d, e, f, g, h, i):
    return a.div_(b).mul_(c) \
            .div_(d).mul_(e) \
            .div_(f).mul_(g) \
            .div_(h).mul_(i)


def muldivabb(a, b, c, d, e):
    return a.mul(b).div(b) \
            .mul(c).div(c) \
            .mul(d).div(d) \
            .mul(e).div(e)

def divmulabb(a, b, c, d, e):
    return a.div(b).mul(b) \
            .div(c).mul(c) \
            .div(d).mul(d) \
            .div(e).mul(e) 

def addsubabb(a, b, c, d, e):
    return a.add(b).sub(b) \
            .add(c).sub(c) \
            .add(d).sub(d) \
            .add(e).sub(e) 


def subaddabb(a, b, c, d, e):
    return a.sub(b).add(b) \
            .sub(c).add(c) \
            .sub(d).add(d) \
            .sub(e).add(e) 
