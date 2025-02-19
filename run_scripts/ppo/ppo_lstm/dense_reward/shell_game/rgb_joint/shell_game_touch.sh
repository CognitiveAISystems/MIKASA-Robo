#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 baselines/ppo/ppo_memtasks_lstm.py \
        --env_id=ShellGameTouch-v0 \
        --exp-name=ppo-lstm-dense-shell-game-touch-v0 \
        --capture-video \
        --save-model \
        --track \
        --num-steps=90 \
        --num_eval_steps=90 \
        --include-rgb \
        --include-joints \
        --seed=$seed \
        --total-timesteps=7_500_000 \
        --eval-freq=25 \
        --num-envs=128 \
        --num-minibatches=1 \
        --update-epochs=4 \
        --lstm-hidden-size=512 \
        --lstm-num-layers=1 \
        --lstm-dropout=0.0 \
        --learning-rate=3e-4 \
        --anneal-lr \
        --ent-coef=0.0
done

# n128-feat256-ent0.0-nmb32-3e4-512-1-0.0-
#nmb32 default
