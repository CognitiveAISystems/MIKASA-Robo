#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 baselines/ppo/ppo_memtasks.py \
        --env_id=RotateStrictPos-v0 \
        --exp-name=rotate-strict-pos-v0 \
        --capture-video \
        --save-model \
        --track \
        --num-steps=90 \
        --num_eval_steps=270 \
        --include-state \
        --total_timesteps=50_000_000 \
        --seed=$seed
done