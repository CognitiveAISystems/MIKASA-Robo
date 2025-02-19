#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 baselines/ppo/ppo_memtasks.py \
        --env_id=ShellGameTouch-v0 \
        --exp-name=ppo-mlp-state-dense-shell-game-touch-v0 \
        --capture-video \
        --save-model \
        --track \
        --num-steps=90 \
        --num_eval_steps=90 \
        --include-state \
        --seed=$seed \
        --total-timesteps=20_000_000
done