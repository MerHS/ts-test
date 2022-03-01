import torch
import time

def run(args):
    def fn(a, b):
        r = a.mul(b)
        return r

    device = 'cuda' if args.cuda else 'cpu'
    size = [args.height, args.width]

    # TODO: should match the number of tensors with #param of fn
    # also, with #args of torch.jit.trace and fn
    a = torch.rand(size, device=device)
    b = torch.rand(size, device=device)

    trace_time = 0
    if args.ts:
        t1 = time.time()
        fn = torch.jit.trace(fn, (a, b))
        t2 = time.time()
        trace_time = t2 - t1

    # warm-up 1000 times
    for _ in range(args.warm):
        r = fn(a, b)
        r.storage()

    t1 = time.time()
    for _ in range(args.iter):
        r = fn(a, b)
        r.storage()

    t2 = time.time()
    exec_time = t2 - t1

    return trace_time, exec_time