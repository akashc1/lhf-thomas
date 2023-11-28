#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Initialize data for training the reward function."""

import torch
from tqdm import tqdm 

def init_data(generator, embedder, oracle, config):
    r"""Initialize data for training the reward function.

    We only append the *feature* of the query (instead of
    the query itself) to the buffer to save memory. At the
    end of the day, the buffer is used to train the reward
    model, which only needs the feature of the query.

    To query the oracle, we need the direct output from
    generator.generate (instead of the embedded version) since
    the oracle need to convert generator vocab --> string -->
    oracle vocab.
    """
    e_queries = []
    prefers = []
    prompts_queries = []

    if config.batch_size*config.n_init_data//10 == 0:
        num_iter = 1
    else:
        num_iter = config.batch_size*config.n_init_data//10

    for _ in tqdm(range(num_iter)):

        if config.max_length == 2 and config.exp_name == "SYN":
            prompt = torch.randint(
                15 * config.vocab_size_generator//20,
                20 * config.vocab_size_generator//20,
                (1, 10), device=config.device)
            if config.use_dynamic_gradient:
                prompt = torch.nn.functional.one_hot(
                    prompt, num_classes=config.vocab_size_generator
                ).float()
        else:
            prompt = oracle.get_prompt(
                batch_size=10,
                prompt_length=config.prompt_length
            )
            # >>> prompt_len x batch_size x num_categories_generator
        outputs, *_ = generator.generate(
            input_tensors=prompt,
            n_restart=1,
            q=config.q,
            max_length=config.max_length,
        )
        # >>> batch_size x (n_restart=1) x q
        # ... x max_length (x vocab_size_generator)

        outputs = outputs.squeeze(1)
        prompts_queries.append(outputs.detach())
        # >>> batch_size x q x max_length (x vocab_size_generator)

        prefers.append(oracle(outputs).detach())
        # e_queries.append(
        #     embedder(
        #         sentence=outputs,
        #         g2e_transl=generator.g2e_transl,
        #     ).detach()
        # )
        # >>> batch_size x q x embed_dim

    prompts_queries = torch.cat(prompts_queries, dim=0)
    # >>> n_init_data x q x max_length (x vocab_size_generator)
    
    # e_queries = torch.cat(e_queries, dim=0)
    # >>> n_init_data x q x embed_dim

    prefers = torch.cat(prefers, dim=0)
    # >>> n_init_data
    
    return dict(prompts_queries=prompts_queries, 
                # e_queries=e_queries, 
                prefers=prefers)
