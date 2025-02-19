#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 baselines/ppo/ppo_memtasks.py \
        --env_id=InterceptGrabSlow-v0 \
        --exp-name=ppo-mlp-state-dense-intercept-grab-slow-v0 \
        --capture-video \
        --save-model \
        --track \
        --num-steps=90 \
        --num_eval_steps=270 \
        --include-state \
        --total_timesteps=35_000_000 \
        --eval_freq=50 \
        --seed=$seed \
        --anneal-lr

done