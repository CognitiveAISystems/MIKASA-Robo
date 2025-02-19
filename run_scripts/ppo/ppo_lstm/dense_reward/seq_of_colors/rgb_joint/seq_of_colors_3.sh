#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 baselines/ppo/ppo_memtasks_lstm.py \
        --env_id=SeqOfColors3-v0 \
        --exp-name=ppo-lstm-rgb-joint-dense-seq-of-colors-3-v0 \
        --capture-video \
        --save-model \
        --track \
        --num-steps=120 \
        --num_eval_steps=360 \
        --include-rgb \
        --include-joints \
        --seed=$seed \
        --total-timesteps=50_000_000 \
        --eval-freq=55 \
        --num-envs=256 \
        --num-minibatches=8 \
        --update-epochs=4 \
        --lstm-hidden-size=512 \
        --lstm-num-layers=1 \
        --lstm-dropout=0.0 \
        --learning-rate=3e-4 \
        --anneal-lr
        
done