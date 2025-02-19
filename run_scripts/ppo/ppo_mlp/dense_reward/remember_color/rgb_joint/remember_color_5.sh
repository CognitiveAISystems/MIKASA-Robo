#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 baselines/ppo/ppo_memtasks.py \
        --env_id=RememberColor5-v0 \
        --exp-name=ppo-mlp-dense-remember-color-5-v0 \
        --capture-video \
        --save-model \
        --track \
        --num-steps=60 \
        --num-eval-steps=180 \
        --include-rgb \
        --include-joints \
        --seed=$seed \
        --total-timesteps=20_000_000 \
        --no-finite-horizon-gae \
        --eval-freq=50 \
        --num-envs=256 \
        --num-minibatches=8 \
        --gae-lambda=0.9 \
        --gamma=0.8
done