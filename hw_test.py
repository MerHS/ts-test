import torch
import argparse
from tester import test_fn, test_fn_inplace
from simple import hw_fns as fns
from inspect import signature

torch.manual_seed(0)

parser = argparse.ArgumentParser("benchmark")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--ts", action="store_true")
parser.add_argument("--ts_script", action="store_true")
parser.add_argument("--fuse", action="store_true")
parser.add_argument("--nnc", action="store_true")
parser.add_argument("--nvfuse", action="store_true")
parser.add_argument("--grad", action='store_true')
parser.add_argument("--iter_n", type=int, default=10000)
parser.add_argument("--warm", type=int, default=1000)

args = parser.parse_args()

iter_n = args.iter_n
cuda = False
torch._C._jit_set_texpr_fuser_enabled(False)

if args.cuda:
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

tensor_pool = list()
len_list = [80, 160, 320, 640, 1280]
for height in len_list:
    for width in len_list:
        size = [height, width]
        tensors = [torch.rand(size, device=device) for _ in range(16)]

        tensor_pool.append((size, tensors))

print(args)

def run_test():
    for fn_name in fns.__dict__.keys():
        if fn_name.startswith("_") or fn_name == "torch":
            continue
        print(fn_name)
        for (hw, tensors) in tensor_pool:
            fn = fns.__dict__[fn_name]
            fn_sig = signature(fn)
            param_len = len(fn_sig.parameters)
            params = tensors[:param_len]
            params[0] = params[0].clone()

            n = iter_n
            if fn_name.startswith("x_"):
                n = iter_n // 20

            run_time = test_fn(fn, args, params, n)

            print(f"{run_time:8.5f},", end='')

            if hw[1] == len_list[len(len_list) - 1]:
                print(fn_name, flush=True)

if args.grad:
    with torch.enable_grad():
        run_test()
else:
    with torch.no_grad():
        run_test()
    

