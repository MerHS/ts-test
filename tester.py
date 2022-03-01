import torch
import time


def test_fn(fn, args, tensors, iter_n):
    if args.ts:
        fn = torch.jit.trace(fn, tensors)

    for _ in range(args.warm):
        r = fn(*tensors)
        r.storage()

    t1 = time.time()
    for _ in range(iter_n):
        r = fn(*tensors)
        r.storage()
    t2 = time.time()

    return t2 - t1


def test_fn_inplace(fn, args, tensors, iter_n):
    first = tensors[0].clone()

    if args.ts:
        fn = torch.jit.trace(fn, tensors)

    for _ in range(args.warm):
        r = fn(*tensors)
        r.storage()

    t1 = time.time()
    for _ in range(iter_n):
        params[0] = first.clone()
        r = fn(*tensors)
        r.storage()
    t2 = time.time()

    return t2 - t1
