#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 baselines/ppo/ppo_memtasks.py \
        --env_id=InterceptSlow-v0 \
        --exp-name=ppo-mlp-dense-intercept-slow-v0 \
        --capture-video \
        --save-model \
        --track \
        --num-steps=90 \
        --num_eval_steps=270 \
        --include-rgb \
        --include-joints \
        --seed=$seed \
        --total-timesteps=10_000_000 \
        --eval-freq=25 \
        --num-envs=256 \
        --num-minibatches=32
done