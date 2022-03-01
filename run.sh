#!/bin/bash

echo "fn,height,width,trace_time,exec_time,total_time,cuda,torchscript,fuse,nnc,nvfuse"

N=10000
CN=50000
for fn in add mul addmul muladd add8 mul8 addmul8
do
for h in 80 160 320 640 1280
do
    for w in 80 160 320 640 1280
    do
        python runner.py --fn=$fn --iter=$N --warm=100  --height=$h --width=$w
        python runner.py --fn=$fn --iter=$N --warm=100 --ts  --height=$h --width=$w
        python runner.py --fn=$fn --iter=$N --warm=100 --ts --fuse  --height=$h --width=$w
        python runner.py --fn=$fn --iter=$N --warm=100 --ts --fuse --nnc  --height=$h --width=$w
        python runner.py --fn=$fn --iter=$CN --warm=100 --ts --cuda  --height=$h --width=$w
        python runner.py --fn=$fn --iter=$CN --warm=100 --ts --cuda --fuse  --height=$h --width=$w
        python runner.py --fn=$fn --iter=$CN --warm=100 --ts --cuda --fuse --nnc  --height=$h --width=$w
        python runner.py --fn=$fn --iter=$CN --warm=100 --ts --cuda --fuse --nvfuse  --height=$h --width=$w
    done
done
done
