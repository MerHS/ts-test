import torch
import sys
import time
import argparse

tx = time.time()

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
parser.add_argument("-n", type=int, default=1000)

args = parser.parse_args()

if args.torchy:
  import torchy
  torchy.enable()
if args.cuda:
  cuda = True
  if not torch.cuda.is_available():
    print('UNSUPPORTED: CUDA is not available')
    exit(0x42)
if args.ts:
  torchscript = True
if args.fuse:
  torch._C._jit_set_texpr_fuser_enabled(True)
if args.nnc:
  torch._C._jit_set_texpr_reductions_enabled(True)
if args.nvfuse:
  #os.environ['PYTORCH_CUDA_FUSER_DISABLE_FALLBACK'] = '1'
  #os.environ['PYTORCH_CUDA_FUSER_DISABLE_FMA'] = '1'
  #os.environ['PYTORCH_CUDA_FUSER_JIT_OPT_LEVEL'] = '0'
  torch._C._jit_set_nvfuser_enabled(True)

device = 'cuda' if cuda else 'cpu'
size = [1000,1000]

print('====rand====')

a = torch.rand(size, device=device)
b = torch.rand(size, device=device)
c = torch.rand(size, device=device)

def fn(a, b, c):
  r = a.add(b).mul(c)
  return r

print('====trace====')

if torchscript:
  fn = torch.jit.trace(fn, (a, b, c))

t = time.time()
print(f'====init time {t - tx}====')

for _ in range(args.n):
  r = fn(a, b, c)
  r.storage()

print(f'====running time {time.time() - t}====')

if hasattr(fn, 'graph'):
  print(fn.graph)