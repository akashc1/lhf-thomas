#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Naive BO acquisition function."""

import torch
import matplotlib.pyplot as plt

# Generator -> generate  q completions for each prompt
# Select the best completion for each prompt by using the reward model
# Train the generator to generate the best completions
# Loss: NLL of the best completions and the generated completions

def naive_bo(
    config,
    generator,
    embedder,
    reward_model,
    oracle,
    generator_optim,
    iteration,
    *args,
    **kwargs,
):
    acqf_loss = []
    for i in range(config.generator_iter):
        # Get prompt from user and reply
        prompt = oracle.get_prompt(
            batch_size=config.batch_size,
            prompt_length=config.prompt_length,
        )
        # >>> prompt_len x batch_size x vocab_size_generator

        #### Generate completions for choosing best
        outputs, outputs_probs, *_ = generator.generate(
            input_tensors=prompt,
            n_restart=config.n_restart,
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

        rewards = reward_model.posterior_function(embed_outputs).squeeze(-1)
        # >>> n_samples x batch_size x n_restart x q

        rewards = rewards.mean(0)
        # >>> batch_size x n_restart x q
        
        best_reward_idx = rewards.argmax(-1)
        # >>> batch_size x n_restart

        batch_idx = torch.arange(config.batch_size)
        batch_idx = batch_idx.unsqueeze(-1).expand_as(best_reward_idx)
        restart_idx = torch.arange(config.n_restart)
        restart_idx = restart_idx.unsqueeze(0).expand_as(best_reward_idx)
        best_completions = outputs_probs[batch_idx, restart_idx, best_reward_idx].detach()
        # >>> batch_size x n_restart
        # ... x max_length x vocab_size_generator
        
        best_completions = best_completions[:, :, None].expand_as(outputs_probs)
        # >>> batch_size x n_restart x q
        # ... x max_length x vocab_size_generator
        #### Done generating completionss
        
        #### Calculate loss between generated completions and the best completions
        losses = torch.distributions.categorical.Categorical(outputs_probs)
        losses = - losses.log_prob(best_completions.argmax(-1))
        # >>> batch_size x n_restart x q x max_length
        
        losses = losses.mean((-1, -2))
        # >>> batch_size x n_restart

        loss = losses.mean()
        loss.backward()
        generator_optim.step()
        generator_optim.zero_grad()
        acqf_loss.append(loss.item())
        if i % 100 == 0:
            print(f"Acqf loss {i}/{config.generator_iter}: {loss.item():.3f}")

    # select pair with lowest loss
    i = losses.argmin(-1)
    j = torch.arange(config.batch_size)
    # >>> batch_size

    plt.figure()
    plt.plot(acqf_loss)
    plt.savefig(f"results/{config.exp_id}/acqf_loss{iteration}.pdf")
    plt.close()
    
    return outputs[j, i], embed_outputs[j, i]
