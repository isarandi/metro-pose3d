## Training

Make sure to [install the dependencies](DEPENDENCIES.md) and [prepare the datasets](DATASETS.md) first. 

After that, you can train the network. Some example configurations:

```bash
$ cd config
$ ../src/main.py \
    --train --dataset=h36m --train-on=trainval --epochs=27 --seed=1 \
    --logdir=h36m/metro_seed1 --print-log --gui
$ ../src/main.py \
    --train --dataset=h36m --train-on=trainval --epochs=27 --seed=1 \
    --scale-recovery=bone-lengths \
    --logdir=h36m/2.5d_seed1 --print-log --gui

$ ../src/main.py \
    --train --dataset=mpi-inf-3dhp --train-on=trainval --epochs=27 --seed=1 \
    --universal-skeletons --logdir=h36m/metro_univ_seed1 --print-log --gui
$ ../src/main.py \
    --train --dataset=mpi-inf-3dhp --train-on=trainval --epochs=27 --seed=1 \
    --no-universal-skeletons --logdir=h36m/metro_nonuniv_seed1 --print-log --gui
```