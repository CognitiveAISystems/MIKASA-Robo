#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 baselines/ppo/ppo_memtasks.py \
        --env_id=RememberShape3-v0 \
        --exp-name=ppo-mlp-sparse-remember-shape-3-v0 \
        --capture-video \
        --save-model \
        --track \
        --num-steps=60 \
        --num_eval_steps=180 \
        --include-rgb \
        --include-joints \
        --seed=$seed \
        --total-timesteps=20_000_000 \
        --eval-freq=25 \
        --num-envs=256 \
        --num-minibatches=8 \
        --reward-mode=sparse \
        --learning-rate=1e-4 \
        --ent-coef=0.1 \
        --clip-coef=0.1 \
        --update-epochs=8

done
