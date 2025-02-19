#!/bin/bash

for seed in 123 231 321
do
    echo "Running experiment with seed $seed"
    python3 baselines/ppo/ppo_memtasks.py \
        --env_id=TakeItBack-v0 \
        --exp-name=ppo-mlp-state-dense-take-it-back-v0 \
        --capture-video \
        --save-model \
        --track \
        --num-steps=180 \
        --num_eval_steps=540 \
        --include-state \
        --seed=$seed \
        --update-epochs=4 \
        --num-envs=4096 \
        --num-minibatches=32 \
        --target-kl=0.1 \
        --total_timesteps=200_000_000 \
        --eval-freq=25 \
        --learning-rate=3e-4 \
        --anneal-lr
done

# batch_size = 16384 (737280)
# minibatch_size = 512 (23040)
# num_iterations = 3051 (67)
# num_steps = 4 (90)

"""
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
"""