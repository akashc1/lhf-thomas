#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Train reward model with preference data."""

from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch

def train_reward(
    config, generator, embedder, reward_model,
    reward_optim, reward_scaler, buffer, iteration
):
    r"""Train reward model with preference data."""
    if config.reset_reward:
        if config.ABC_method == "MCD":
            for layer in reward_model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        elif config.ABC_method == "Ensemble":
            for submodel in reward_model.children():
                for layer in submodel.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                
        reward_optim = torch.optim.Adam(
            reward_model.parameters(),
            lr=config.reward_lr,
        )
        
    e_queries = buffer["prompts_queries"]
    e_queries = embedder(
                        sentence=e_queries,
                        g2e_transl=generator.g2e_transl,
                )
    losses = []
    for i in range(config.reward_iter):
        reward_optim.zero_grad()
        # with torch.cuda.amp.autocast():
        dist = Categorical(logits=reward_model(e_queries))
        # >>> [n_samples] x batch_size x q

        if config.ABC_method == "MCD":
            label = buffer["prefers"]
        elif config.ABC_method == "Ensemble":
            label = buffer["prefers"][None].repeat(config.n_samples, 1)
        else:
            raise NotImplementedError
        
        loss = -dist.log_prob(label).mean()

        # reward_scaler.scale(loss).backward()
        # reward_scaler.unscale_(reward_optim)
        # torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=2.0)
        # reward_scaler.step(reward_optim)
        # reward_scaler.update()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=2.0)
        reward_optim.step()
            
        losses.append(loss.item())
        if i % (config.reward_iter // 5) == 0:
            print(f"Reward loss {i}/{config.reward_iter}: {loss.item():.3f}")

    plt.figure()
    plt.plot(losses)
    plt.savefig(f"results/{config.exp_id}/reward_loss{iteration}.pdf")
    plt.close()
