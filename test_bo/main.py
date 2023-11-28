#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement an active learning loop for a LLM."""

import os
import torch
import pickle
import loralib as lora

from configs import Config
from my_utils.decorator import FreezeParameters
from init_agent import init_agent
from init_data import init_data
from train_reward import train_reward
from my_utils.eval_func import eval_func
import torch.multiprocessing as mp
import torch.distributed as distributed

def main(args):
    for seed in args.seeds:
         # Init config, agent, buffer
        config = Config(args, seed)
        
        # mp.spawn(__main__,
        #         args=(config,),
        #         nprocs=config.nprocs,
        #         join=True
        # )
        __main__(0, config)
        
def __main__(rank, config):
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '29500'
        # distributed.init_process_group("gloo", rank=rank, world_size=config.nprocs)

        (
            embedder,
            generator,
            oracle,
            reward_model,
            generator_optim,
            reward_optim,
            generator_scaler,
            reward_scaler,
            reward_output_scaler
        ) = init_agent(rank, config)
        
        with torch.no_grad():
            buffer = init_data(
                config=config,
                generator=generator,
                embedder=embedder,
                oracle=oracle,
            )

        regrets = []
        scores = []

        for iteration in range(0, config.bo_iter+1):
            print(
                "*" * 20,
                f"Iteration {iteration}/{config.bo_iter}",
                "*" * 20
            )

            with torch.no_grad():
                eval_func(
                    config=config,
                    generator=generator,
                    embedder=embedder,
                    reward_model=reward_model,
                    oracle=oracle,
                    buffer=buffer,
                    regrets=regrets,
                    scores=scores,
                    iteration=iteration,
                )

            # lora.mark_only_lora_as_trainable(reward_model)
            train_reward(
                config=config,
                generator=generator,
                embedder=embedder,
                reward_model=reward_model,
                reward_optim=reward_optim,
                reward_scaler=reward_scaler,
                buffer=buffer,
                iteration=iteration,
            )
            
            with FreezeParameters([reward_model]):      
                # lora.mark_only_lora_as_trainable(generator, bias='lora_only')         
                outputs, embed_outputs = config.acqf(
                    config=config,
                    generator=generator,
                    embedder=embedder,
                    reward_model=reward_model,
                    reward_output_scaler=reward_output_scaler,
                    oracle=oracle,
                    generator_optim=generator_optim,
                    generator_scaler=generator_scaler,
                    buffer=buffer,
                    iteration=iteration,
                )
            
            buffer["prefers"] = torch.cat(
                [buffer["prefers"], oracle(outputs).detach()], dim=0
            )
            buffer["prompts_queries"] = torch.cat(
                [buffer["prompts_queries"], outputs.detach()], dim=0
            )
            # buffer["e_queries"] = torch.cat(
            #     [buffer["e_queries"], embed_outputs.detach()], dim=0
            # )

            # Save metrics and buffer
            save_dict = dict(regret=regrets, score=scores, buffer=buffer)
            for key, value in save_dict.items():
                pickle.dump(value, open(f"results/{config.exp_id}/{key}.pkl", "wb"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="SynGP")
    parser.add_argument("--exp_id", type=str, default="0")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max_length", type=int, default=2)
    parser.add_argument("--reward_iter", type=int, default=100)
    parser.add_argument("--vocab_size", type=int, default=180)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--ABC_method", type=str, default="Ensemble")
    parser.add_argument("--reset_reward", action="store_true")
    parser.add_argument("--algo", type=str, default="bo")
    args = parser.parse_args()
    
    main(args)
