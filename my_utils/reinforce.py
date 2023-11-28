import torch
import matplotlib.pyplot as plt


def reinforce(
    config,
    generator,
    embedder,
    reward_model,
    oracle,
    generator_optim,
    generator_scaler,
    iteration,
    *args,
    **kwargs,
):
    
    rl_loss = []
    for i in range(config.generator_iter):
        # Get prompt from user and reply
        prompt = oracle.get_prompt(
            batch_size=config.batch_size,
            prompt_length=config.prompt_length,
        )
        # >>> prompt_len x batch_size x vocab_size_generator

        outputs, outputs_probs, entropies, log_probs = generator.generate(
            input_tensors=prompt,
            n_restart=1,
            q=config.q,
            max_length=config.max_length, 
        )
        # >>> batch_size x n_restart x q x max_length x num_categories

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

        batch_size = outputs.shape[0]
        generated_length = config.max_length - config.prompt_length
        reward = torch.zeros(*outputs.shape[:3], generated_length).to(outputs.device)
        # >>> batch_size x n_restart x q x generated_length

        posterior_sample = reward_model.posterior_function(embed_outputs).squeeze(-1)
        # >>> n_samples x batch_size x n_restart x q
        
        reward_mean = posterior_sample.mean(0).detach()
        # >>> batch_size x n_restart x q

        reward[..., -1] = reward_mean
        # >>> batch_size x generated_length

        # compute discount reward
        for t in range(reward.shape[-1]-1):
            reward[..., -t-2] = reward[..., -t-1] * config.discount
            
        # compute loss
        losses = -log_probs * reward
        losses = losses - config.entropy_weight * entropies
        # >>> batch_size x generated_length

        loss = losses.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        generator_optim.step()
        generator_optim.zero_grad()
        rl_loss.append(loss.item())
    
        if i % 20 == 0:
            print(f"RL loss {i}/{config.generator_iter}: {loss.item():.3f}")

    plt.figure()
    plt.plot(rl_loss)
    plt.savefig(f"results/{config.exp_id}/rl_loss{iteration}.pdf")
    plt.close()

    return outputs[:, 0], embed_outputs[:, 0]