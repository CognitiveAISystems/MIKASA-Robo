#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 baselines/ppo/ppo_memtasks.py \
        --env_id=RememberColor3-v0 \
        --exp-name=ppo-mlp-sparse-remember-color-3-v0 \
        --capture-video \
        --save-model \
        --track \
        --num-steps=60 \
        --num-eval-steps=180 \
        --include-rgb \
        --include-joints \
        --seed=$seed \
        --total-timesteps=10_000_000 \
        --no-finite-horizon-gae \
        --eval-freq=50 \
        --num-envs=256 \
        --reward-mode=sparse
done