#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Optimize acquisition function."""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from my_utils.decorator import FreezeParameters
from my_utils.cat_with_base import OneHotCategoricalSTVarianceReduction as OHCTVR


def optimize_acqf(
    config,
    generator,
    embedder,
    reward_model,
    prompt,
    generator_optim,
    iteration,
):
    temperature = 0.1
    theta = torch.normal(0,5,
        size = (config.vocab_size_generator,
        config.vocab_size_generator),
        requires_grad=True,
        device=config.device,
    )
    generator_optim = torch.optim.Adam([theta], lr=1e-2)
    vec = torch.arange(config.vocab_size_generator, device=config.device)

    sm = torch.nn.Softmax(dim=-1)
    with FreezeParameters([reward_model]):
        acqf_loss = []
        for i in range(config.acqf_iter):
            prompt = torch.randint(
                0,
                config.vocab_size_generator,
                (100, 1),
                device=config.device,
            )
            # >>> 100 x 1

            prompt = F.one_hot(prompt, config.vocab_size_generator).float()
            # >>> 100 x 1 x vocab_size_generator

            probs = prompt @ theta
            # >>> 100 x 1 x vocab_size_generator

            probs = sm(probs)
            # >>> 100 x 1 x vocab_size_generator
            
            dist = OHCTVR(probs=probs)

            outputs = dist.rsample((100,))
            # >>> 1000 x 100 x 1 x vocab_size_generator

            outputs = torch.cat([prompt[None].expand_as(outputs), outputs], dim=2)        
            # >>> 1000 x 100 x 2 x vocab_size_generator

            outputs = (outputs * vec).sum(-1)

            # rewards = abs(outputs).sum(-1)

            rewards = -abs(outputs[0] - outputs[1])

            # embed_outputs = embedder(
            #     sentence=outputs,
            #     g2e_transl=generator.g2e_transl,
            # )
            # print(embed_outputs)
            # >>> 1000 x 100 x 2

            # rewards = reward_model.posterior_function(embed_outputs, n_samples=2)
            # >>> 64 x 1000 x 100 x 1

            loss = -rewards.mean() #- temperature * dist.entropy().mean()

            loss.backward()
            generator_optim.step()
            generator_optim.zero_grad()
            acqf_loss.append(loss.item())
            if i % 20 == 0:
                print(f"Acqf loss {i}/{config.acqf_iter}: {loss.item():.3f}")
                print(sm(theta).cpu().detach().numpy()[6])

    # draw a heatmap of theta with colorbar
    plt.figure()
    plt.imshow(sm(theta).T.cpu().detach().numpy())
    plt.colorbar()
    plt.savefig(f"results/{config.exp_id}/theta{iteration}.pdf")
    plt.close()


    # # select pair with highest reward according to utility
    # i = losses.argmax(-1)
    # j = torch.arange(config.batch_size)
    # # >>> batch_size

    plt.figure()
    plt.plot(acqf_loss)
    plt.savefig(f"results/{config.exp_id}/acqf_loss{iteration}.pdf")
    plt.close()

    breakpoint()

    return outputs[j, i], embed_outputs[j, i]
