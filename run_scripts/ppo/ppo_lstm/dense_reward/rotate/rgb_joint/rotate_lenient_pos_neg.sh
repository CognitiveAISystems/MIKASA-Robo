#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 baselines/ppo/ppo_memtasks_lstm.py \
        --env_id=RotateLenientPosNeg-v0 \
        --exp-name=v2-ppo-lstm-dense-rotate-lenient-pos-neg-v0 \
        --capture-video \
        --save-model \
        --track \
        --num-steps=90 \
        --num_eval_steps=270 \
        --include-rgb \
        --include-joints \
        --seed=$seed \
        --total-timesteps=20_000_000 \
        --num-envs=512 \
        --num-minibatches=1 \
        --update-epochs=4 \
        --lstm-hidden-size=512 \
        --lstm-num-layers=1 \
        --lstm-dropout=0.0 \
        --learning-rate=3e-4 \
        --anneal-lr
done