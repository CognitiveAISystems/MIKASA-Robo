#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 baselines/ppo/ppo_memtasks_lstm.py \
        --env_id=RememberColor5-v0 \
        --exp-name=ppo-lstm-sparse-remember-color-5-v0 \
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
        --num-envs=128 \
        --num-minibatches=1 \
        --update-epochs=4 \
        --lstm-hidden-size=512 \
        --lstm-num-layers=1 \
        --lstm-dropout=0.0 \
        --learning-rate=3e-4 \
        --anneal-lr \
        --reward-mode=sparse
done