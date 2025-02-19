#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 baselines/ppo/ppo_memtasks_lstm.py \
        --env_id=RememberShapeAndColor3x2-v0 \
        --exp-name=ppo-lstm-dense-remember-shape-and-color-3x2-v0 \
        --capture-video \
        --save-model \
        --track \
        --num-steps=60 \
        --num_eval_steps=180 \
        --include-rgb \
        --include-joints \
        --seed=$seed \
        --total-timesteps=5_000_000 \
        --no-finite-horizon-gae \
        --eval-freq=25 \
        --gae-lambda=0.9 \
        --gamma=0.8 \
        --num-envs=256 \
        --num-minibatches=8 \
        --update-epochs=4 \
        --lstm-hidden-size=512 \
        --lstm-num-layers=1 \
        --lstm-dropout=0.0 \
        --learning-rate=3e-4 \
        --anneal-lr
        
done