#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Optimize acquisition function."""

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
    
def optimize_acqf(
    config,
    generator,
    embedder,
    buffer,
    reward_model,
    reward_output_scaler,
    oracle,
    generator_optim,
    generator_scaler,
    iteration,
    *args,
    **kwargs,
):
    acqf_loss = []
    if config.use_TS:
        TS_idx = torch.randint(high=config.n_samples, size=(1,))
        TS_idx = TS_idx.item()
        print("IDX", TS_idx)
    else:
        TS_idx = None
    
    # batch_size = config.batch_size
    batch_size = 100

    for i in tqdm(range(config.generator_iter)):
        generator_optim.zero_grad()
        # with torch.cuda.amp.autocast():
        # Get prompt from user and reply
        prompt = oracle.get_prompt(
            # batch_size=config.batch_size,
            batch_size=batch_size,
            prompt_length=config.prompt_length,
        )
        # >>> prompt_len x batch_size x vocab_size_generator
        
        outputs, outputs_probs, _, log_probs = generator.generate(
            input_tensors=prompt,
            n_restart=config.n_restart,
            q=config.q,
            max_length=config.max_length,
        )
        # >>> batch_size x n_restart x q
        # ... x max_length [x vocab_size_generator]

        if config.use_dynamic_gradient:
            embed_outputs = embedder(
                sentence=outputs_probs,
                g2e_transl=generator.g2e_transl,
            )
            # >>> batch_size x n_restart x q x embed_dim
            
            losses = - reward_model.posterior_function(embed_outputs).squeeze(-1)
            # >>> n_samples x batch_size x n_restart x q
            entropies_q = 0
            
        else:
            embed_outputs = embedder(
                sentence=outputs,
                g2e_transl=generator.g2e_transl,
            )
            # >>> batch_size x n_restart x q x embed_dim
            
            reward = reward_model.posterior_function(
                input=embed_outputs, TS_idx=TS_idx,
            ).squeeze(-1).detach()
            reward = reward_output_scaler(reward)
            # >>> [n_samples|¬use_TS] x batch_size x n_restart x q

            log_probs = log_probs.sum(-1)
            # >>> batch_size x n_restart x q
            
            if not config.use_TS:
                log_probs = log_probs[None, ...].expand_as(reward)
                # >>> n_samples x batch_size x n_restart x q

            losses = - reward * log_probs
            # >>> [n_samples|¬use_TS] x batch_size x n_restart x q

            # Entropy between q options
            entropies_q = torch.distributions.categorical.Categorical(logits=log_probs).entropy()
            # >>> n_samples x batch_size x n_restart
        
        losses = losses.mean(-1) - entropies_q
        # losses = - losses.max(-1).values - entropies
        # losses = - (losses.softmax(-1)*losses).sum(-1) - entropies
        # >>> [n_samples|¬use_TS] x batch_size x n_restart

        if config.use_dynamic_gradient or not config.use_TS:
            losses = losses.mean(0)
            # >>> batch_size x n_restart

        loss = losses.mean()

        # generator_scaler.scale(loss).backward()
        # generator_scaler.step(generator_optim)
        # generator_scaler.update()
        loss.backward()
        generator_optim.step()
        acqf_loss.append(loss.item())
        if i % (config.generator_iter//5) == 0:
            print(f"Acqf loss {i}/{config.generator_iter}: {loss.item():.3f}")

    # select pair with lowest loss
    i = losses.argmin(-1)
    j = torch.arange(batch_size)
    # >>> batch_size

    plt.figure()
    plt.plot(acqf_loss)
    plt.savefig(f"results/{config.exp_id}/acqf_loss{iteration}.pdf")
    plt.close()
    
    return outputs[j, i], embed_outputs[j, i]
