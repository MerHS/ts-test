import torch
import time
from inspect import signature

def test_fn(fn, args, tensors, iter_n):
    fn_sig = signature(fn)
    param_len = len(fn_sig.parameters)
    params = tensors[:param_len]

    if args.ts:
        fn = torch.jit.trace(fn, param_len)

    for _ in range(args.warm):
        r = fn(*params)
        r.storage()

    t1 = time.time()
    for _ in range(iter_n):
        r = fn(*params)
        r.storage()
    t2 = time.time()

    return t2 - t1    

def test_fn_inplace(fn, args, tensors, iter_n):
    fn_sig = signature(fn)
    param_len = len(fn_sig.parameters)
    params = tensors[:param_len]
    first = params[0].clone()

    if args.ts:
        fn = torch.jit.trace(fn, param_len)

    for _ in range(args.warm):
        r = fn(*params)
        r.storage()

    t1 = time.time()
    for _ in range(iter_n):
        params[0] = first.clone()
        r = fn(*params)
        r.storage()
    t2 = time.time()

    return t2 - t1

