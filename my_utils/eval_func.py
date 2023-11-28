#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r""""""

from my_utils.plot import plot, plot_posterior_samples
import matplotlib.pyplot as plt
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())
import torch
from tqdm import tqdm

def eval_func(
    config,
    generator,
    embedder,
    reward_model,
    oracle,
    buffer,
    regrets,
    scores,
    iteration,
):
    outputs = []
    predictions = []
    prefers = []
    local_scores = []
    n_test_sample = 1000
    if n_test_sample // config.batch_size > 0:
        n_iter = n_test_sample // config.batch_size
    else:
        n_iter = 1
        
    for _ in tqdm(range(n_iter)):
        prompt = oracle.get_prompt(
            batch_size=config.batch_size,
            prompt_length=config.prompt_length
        )
        # >>> prompt_len x batch_size x vocab_size_generator
        
        output, *_ = generator.generate(
            input_tensors=prompt,
            n_restart=1,
            q=config.q,
            max_length=config.max_length,
        )
        # >>> batch_size x n_restart x q x max_length [x vocab_size_generator]

        output = output.detach()
        # >>> batch_size x n_restart x q x max_length [x vocab_size_generator]

        output = output.reshape(-1, *output.shape[2:])
        # >>> (batch_size*n_restart) x q x max_length [x vocab_size_generator]

        prefers.append(oracle(output))
        # >>> batch_size*n_restart
        
        output_scores = oracle.compute_score(output)
        # >>> batch_size*n_restart x q

        local_scores.append(output_scores.mean(-1))
        # >>> (batch_size*n_restart)

        embedded_output = embedder(
            sentence=output,
            g2e_transl=generator.g2e_transl,
        )
        # >>> (batch_size*n_restart) x q x embed_dim

        posterior_sample = reward_model.posterior_function(
            embedded_output
        )
        # >>> n_samples x (batch_size*n_restart) x q x 1

        posterior_mean = posterior_sample.squeeze(-1).mean(0)
        # >>> (batch_size*n_restart) x q

        prediction = posterior_mean.argmax(dim=-1)
        # >>> batch_size*n_restart

        predictions.append(prediction)
        outputs.append(output)

    # accuracy of classifier
    outputs = torch.cat(outputs)
    predictions = torch.cat(predictions)
    prefers = torch.cat(prefers)
    local_scores = torch.cat(local_scores)

    regret = (prefers == predictions).float().mean().cpu().detach().item()
    print(f"Classification accuracy: {regret:.3f}")
    print(f"Sample size: {prefers.shape[0]}")

    if config.use_dynamic_gradient:
        outputs = outputs.argmax(dim=-1)
        # >>> (n_restart*batch_size) x q x max_length

    outputs = outputs.reshape(-1, config.max_length).cpu().numpy()

    if config.max_length == 2 and config.exp_name == "SYN":
        plot(
            config=config,
            generator=generator,
            embedder=embedder,
            reward_model=reward_model,
            oracle=oracle,
            name=f"results/{config.exp_id}/posterior{iteration}.pdf",
            e_queries=buffer["prompts_queries"],
            outputs=outputs,
        )
        # plot_posterior_samples(
        #     config=config,
        #     generator=generator,
        #     embedder=embedder,
        #     reward_model=reward_model,
        #     oracle=oracle,
        #     name=f"results/{config.exp_id}/posterior_epochs{config.reward_iter}_{iteration}.pdf",
        #     e_queries=buffer["e_queries"],
        #     outputs=outputs,
        # )

    regrets.append(regret)
    _, ax = plt.subplots(1, 1)
    ax.plot(regrets)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Regret")
    plt.savefig(f"results/{config.exp_id}/regret.pdf")
    plt.close()

    scores.append(local_scores.mean(0).cpu().detach().numpy())
    _, ax = plt.subplots(1, 1) 
    ax.plot(scores)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Score")
    plt.savefig(f"results/{config.exp_id}/score.pdf")
    plt.close()