import torch
import sys
import time
import argparse

torch.manual_seed(0)

cuda = False
torchscript = False

torch._C._jit_set_texpr_fuser_enabled(False)

parser = argparse.ArgumentParser("benchmark")
parser.add_argument("--torchy", action="store_true")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--ts", action="store_true")
parser.add_argument("--fuse", action="store_true")
parser.add_argument("--nnc", action="store_true")
parser.add_argument("--nvfuse", action="store_true")
parser.add_argument("--warm", type=int, default=1000)
parser.add_argument("--iter", type=int, default=1000)
parser.add_argument("--height", type=int, default=1000)
parser.add_argument("--width", type=int, default=1000)
parser.add_argument("--fn", default="basic")

args = parser.parse_args()

if args.cuda:
    cuda = True
    if not torch.cuda.is_available():
        print('UNSUPPORTED: CUDA is not available')
        exit(0x42)
if args.fuse:
    torch._C._jit_set_texpr_fuser_enabled(True)
if args.nnc:
    torch._C._jit_set_texpr_reductions_enabled(True)
if args.nvfuse:
    #os.environ['PYTORCH_CUDA_FUSER_DISABLE_FALLBACK'] = '1'
    #os.environ['PYTORCH_CUDA_FUSER_DISABLE_FMA'] = '1'
    #os.environ['PYTORCH_CUDA_FUSER_JIT_OPT_LEVEL'] = '0'
    torch._C._jit_set_nvfuser_enabled(True)

def b2s(b):
    return "TRUE" if b else "FALSE"

args_str = f'{b2s(args.cuda)},{b2s(args.ts)},{b2s(args.fuse)},{b2s(args.nnc)},{b2s(args.nvfuse)}'

def runner(run):
    trace_time, exec_time = run(args)

    print(f"{args.fn},{args.height},{args.width},{trace_time:8.6f},{exec_time:8.6f},{trace_time + exec_time:8.6f},{args_str}")

if args.fn == 'add':
    import add as fn
    run = fn.run
elif args.fn == 'addmul':
    import addmul as fn
    run = fn.run
elif args.fn == 'mul':
    import mul as fn
    run = fn.run
elif args.fn == 'muladd':
    import muladd as fn
    run = fn.run
elif args.fn == 'add8':
    import add8 as fn
    run = fn.run
elif args.fn == 'mul8':
    import mul8 as fn
    run = fn.run
elif args.fn == 'addmul8':
    import addmul8 as fn
    run = fn.run


runner(run)


