import torch


def finetune_generator(
    config,
    generator,
    buffer,
    generator_optim,
    generator_scaler,
):    
    n_data = buffer["prompts_queries"].shape[0]
    bs = n_data if n_data < 1000 else 1000
    for i in range(config.generator_iter):
        idx = torch.randint(high=n_data, size=(bs,), device=config.device)
        generator_optim.zero_grad()
        samples = buffer["prompts_queries"][idx]
        prefers = buffer["prefers"][idx]
        samples_prefers = samples[idx, prefers]
        # >>> bs x max_length [x vocab_size_generator]
        
        loss = 0
        # training with teacher forcing
        for j in range(config.prompt_length, config.max_length):
            input = samples_prefers[:, :j].transpose(0, 1)
            # >>> max_length x bs [x vocab_size_generator]
            
            target = samples_prefers[:, j]
            # >>> bs [x vocab_size_generator]
            
            dist, _ = generator.compute_next_token_dist(input)
            # >>> bs x vocab_size_generator
            
            loss = loss - dist.log_prob(target).mean()
        
        # generator_scaler.scale(loss).backward()
        # generator_scaler.step(generator_optim)
        # generator_scaler.update()
        loss.backward()
        generator_optim.step()

        if i % (config.generator_iter // 5) == 0:
            print("Generator loss: ", loss.item())
