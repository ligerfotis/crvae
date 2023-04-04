#!/usr/bin/env bash
latent=16
epochs=50
nw=16
optimizer='sgd'
loss='bce'
batch=256
datasets=("MNIST" "EMNIST")
save_integral=1
for beta in 1 0.8 1.2
do
  for seed in 101 202 303 404 505
  do
    for dataset in ${datasets[*]}
    do
      for gamma in 0 0.1 0.5 1 5 10
      do
        for lr in 1e-03
        do
          for model_size in "small" "medium"
          do
            python main.py -d $dataset -aug -si $save_integral --z-dim $latent -bs $batch -e $epochs --beta $beta --gamma $gamma -lr $lr -nw $nw -sh --seed $seed -opt $optimizer -l $loss --model-size $model_size --wandb
          done
        done
      done
    done
  done
done
