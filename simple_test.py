import torch
import argparse
from tester import test_fn, test_fn_inplace
from simple import fns, fns_script
from inspect import signature

ITER = 50000
CUDA_ITER = 50000

torch.manual_seed(0)

parser = argparse.ArgumentParser("benchmark")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--ts", action="store_true")
parser.add_argument("--ts_script", action="store_true")
parser.add_argument("--fuse", action="store_true")
parser.add_argument("--nnc", action="store_true")
parser.add_argument("--nvfuse", action="store_true")
parser.add_argument("--warm", type=int, default=1000)
parser.add_argument("--height", type=int, default=1280)
parser.add_argument("--width", type=int, default=1280)

args = parser.parse_args()

iter_n = ITER
cuda = False
torch._C._jit_set_texpr_fuser_enabled(False)

if args.cuda:
    iter_n = CUDA_ITER
    cuda = True
    if not torch.cuda.is_available():
        print("UNSUPPORTED: CUDA is not available")
        exit(0x42)
if args.fuse:
    torch._C._jit_set_texpr_fuser_enabled(True)
if args.nnc:
    torch._C._jit_set_texpr_reductions_enabled(True)
if args.nvfuse:
    torch._C._jit_set_nvfuser_enabled(True)


device = "cuda" if args.cuda else "cpu"
size = [args.height, args.width]
tensors = [torch.rand(size, device=device) for _ in range(16)]

print(list(fns.__dict__.keys()))

for fn_name in fns.__dict__.keys():
    if fn_name.startswith("_") or fn_name == "torch":
        continue

    fn = fns.__dict__[fn_name]
    fn_sig = signature(fn)
    param_len = len(fn_sig.parameters)
    params = tensors[:param_len]

    if args.ts_script:
        fn = fns_script.__dict__[fn_name]

    # if fn_name.endswith('_'):
    #     run_time = test_fn_inplace(fn, args, tensors, iter_n)
    # else:
    #     run_time = test_fn(fn, args, tensors, iter_n)

    n = iter_n
    if "mm" in fn_name:
        n = iter_n // 20

    run_time = test_fn(fn, args, params, n)

    print(f"{run_time:8.5f}\t{fn_name}", flush=True)
