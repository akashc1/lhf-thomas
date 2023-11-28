#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""FEEDME acquisition function."""

import torch
import matplotlib.pyplot as plt
from finetune_generator import finetune_generator

def feedme(
    config,
    generator,
    embedder,
    buffer,
    reward_model,
    oracle,
    generator_optim,
    generator_scaler,
    iteration,
    *args,
    **kwargs,
):
    
    finetune_generator(
        config=config,
        generator=generator,
        buffer=buffer,
        generator_optim=generator_optim,
        generator_scaler=generator_scaler
    )

    prompt = oracle.get_prompt(
        batch_size=config.batch_size,
        prompt_length=config.prompt_length,
    )
    # >>> prompt_len x batch_size x vocab_size_generator
    
    outputs, outputs_probs, *_ = generator.generate(
        input_tensors=prompt,
        n_restart=1,
        q=config.q,
        max_length=config.max_length,
    )
    # >>> batch_size x n_restart x q
    # ... x max_length x vocab_size_generator
    
    if config.use_dynamic_gradient:
        embed_outputs = embedder(
            sentence=outputs_probs.detach(),
            g2e_transl=generator.g2e_transl,
        )
        # >>> batch_size x n_restart x q x embed_dim
    else:
        embed_outputs = embedder(
            sentence=outputs.detach(),
            g2e_transl=generator.g2e_transl,
        )
        # >>> batch_size x n_restart x q x embed_dim

    rewards = reward_model.posterior_function(embed_outputs).squeeze(-1).mean(0)
    # >>> batch_size x n_restart x q

    rewards = rewards.mean(-1)
    # >>> batch_size x n_restart

    i = rewards.argmax(-1)
    # >>> batch_size
    
    j = torch.arange(config.batch_size)
    # >>> batch_size
    
    return outputs[j, i], embed_outputs[j, i]
