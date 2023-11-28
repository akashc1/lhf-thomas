#!/bin/bash
export PYTHONPATH=$(pwd)

# SYN experiment with the goal of verifying the correctness of the code
#
# Details:
# **** small vocab, short generative length --> 2D picture
# **** large vocab, short generative length --> 2D picture
# **** large vocab, long generative length
# 
# Repeating experiment for:
# **** algo = bo, rl, bon, fe
# **** seed = 12, 13, 14
# **** (max_length, prompt_length, vocab_size_generator)
#    = (2, 1, 20),
#      (2, 1, 60000),
#      (1024, 128, 60000)


# Declare an array of algo, seeds, max_length, vocab_size_generator, gpu_id
algo=(bo nbo rl bon fm)
seeds=(12 13 14)
max_length=(2)
gpu_id=(2 3 4 5 6)

echo "Running experiment"
for m in "${!max_length[@]}"; do
    for s in "${!seeds[@]}"; do
        for i in "${!algo[@]}"; do
            cd test_bo && python main.py \
                    --exp_name SYN \
                    --exp_id _${algo[$i]}_${max_length[$j]}_${seeds[$s]} \
                    --max_length ${max_length[$j]} \
                    --algo ${algo[$i]} \
                    --seeds ${seeds[$s]} \
                    --gpu_id ${gpu_id[$i]} &
        done
        wait
    done
done

for m in "${!max_length[@]}"; do
    for s in "${!seeds[@]}"; do
        cd test_bo && python main.py \
                    --exp_name SYN \
                    --exp_id _bo+dg_${max_length[$j]}_${seeds[$s]} \
                    --max_length ${max_length[$j]} \
                    --algo bo \
                    --seeds ${seeds[$s]} \
                    --dynamic \
                    --gpu_id ${gpu_id[$s]} &
    done
    wait
done

# Plotting
# cd test_bo && python draw_plots.py


# Toy script
# python main.py \
#     --exp_name SYN \
#     --exp_id _test \
#     --max_length 2 \
#     --algo bo \
#     --seeds 13 \
#     --reward_iter 100 \
#     --dynamic \
#     --gpu_id 5

# python main.py \
#     --exp_name SYN \
#     --exp_id _4_epoch_100 \
#     --max_length 2 \
#     --algo bo \
#     --seeds 12 \
#     --reward_iter 100 \
#     --gpu_id 0

# python main.py \
#     --exp_name SYN \
#     --exp_id _4_epoch_500 \
#     --max_length 2 \
#     --algo bo \
#     --seeds 12 \
#     --reward_iter 500 \
#     --gpu_id 1

# python main.py \
#     --exp_name SYN \
#     --exp_id _4_epoch_1000 \
#     --max_length 2 \
#     --algo bo \
#     --seeds 12 \
#     --reward_iter 1000 \
#     --gpu_id 4