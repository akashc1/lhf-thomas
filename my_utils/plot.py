import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tueplots import bundles
plt.rcParams.update(bundles.icml2022())

def plot(
    config,
    generator,
    embedder,
    reward_model,
    oracle,
    name,
    e_queries,
    outputs,
):
    r"""Plot the sample from 2D generator.

    When the design point is in 2D, we can perform
    evaluation by drawing the reward posterior. Did RL
    generate sample that is proportional to the posterior?
    """
    n_panels = config.n_samples
    n_row = int(n_panels**0.5)
    n_col = int(n_panels**0.5)
    fig_size = (20, 20)
    
    if n_panels == 1:
        n_panels = 2
        n_col = 2
        fig_size = (20, 40)
        
    _, ax = plt.subplots(
        int(n_row), int(n_col),
        squeeze=False, figsize=fig_size
    )

    # hidding axis ticks and label
    # and make the plot square
    for i in range(n_panels):
        row = i // n_col
        col = i % n_col 
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
        ax[row, col].set_xlabel("")
        ax[row, col].set_ylabel("")
        ax[row, col].set_aspect("equal")

    vocab_size_generator = oracle.vocab_size_generator
    ploting_points = 30
    levels = 50
    step_size = vocab_size_generator // ploting_points

    
    X = np.arange(0, vocab_size_generator+1, step=step_size)
    Y = np.arange(0, vocab_size_generator+1, step=step_size)
    ploting_points = X.shape[0]
    X[-1] = vocab_size_generator - 1
    Y[-1] = vocab_size_generator - 1
    X, Y = np.meshgrid(X, Y)
    XY = torch.tensor(np.array([X, Y])).to(e_queries.device)
    # >>> 2 x vocab_size_generator x vocab_size_generator

    if config.use_dynamic_gradient:
        # convert XY to one-hot
        XY = F.one_hot(XY, num_classes=vocab_size_generator).float()
        # >>> 2 x vocab_size_generator x vocab_size_generator 
        # ... x vocab_size_generator

    XY = embedder(sentence=XY, g2e_transl=generator.g2e_transl)
    # >>> 2 x vocab_size_generator x vocab_size_generator

    XY = XY.permute(1, 2, 0)
    # >>> vocab_size_generator x vocab_size_generator x 2

    XY = XY.reshape(-1, XY.shape[-1])

    Z_GT = oracle.scorer(XY)
    Z_GT = Z_GT.squeeze(-1).detach().cpu().numpy()
    Z_GT = Z_GT.reshape(ploting_points, ploting_points)
    ax[0, 0].contour(X, Y, Z_GT, levels=levels, cmap="bwr", alpha=0.7)

    Z = reward_model.posterior_function(XY)
    Z_mean = Z.mean(0).squeeze(-1).detach().cpu().numpy()
    Z_mean = Z_mean.reshape(ploting_points, ploting_points)
    ax[0, 1].contourf(X, Y, Z_mean, levels=levels, cmap="bwr", alpha=0.7)

    unique_X = np.unique(X)
    # For each unique x, find the Y that has the highest Z
    # and plot it
    for i in range(unique_X.shape[0]):
        x = unique_X[i]
        idx = np.where(X == x)
        y = Y[idx]
        z = Z_mean[idx]
        y = y[np.argmax(z)]
        ax[0, 1].scatter(x, y, color="green", s=10)

    for i in range(2, n_panels):
        row = i // n_col
        col = i % n_col
        Zi = Z[i].squeeze(-1).detach().cpu().numpy()
        Zi = Zi.reshape(ploting_points, ploting_points)
        ax[row, col].contourf(X, Y, Zi, levels=levels, cmap="bwr", alpha=0.7)

    # mapping the e_queries to the un-embedded queries
    # by undoing the IdentityEmbedder (which do nothing)
    # and the g2e_transl
    # bounds = generator.bounds_embedder
    # range_size = generator.range_size
    # assert torch.any(e_queries >= bounds[0])
    # assert torch.any(e_queries <= bounds[1])
    # queries = (e_queries - bounds[0]) / range_size
    # queries = queries.long()
    # queries[queries == vocab_size_generator] = vocab_size_generator - 1
    # queries = queries.cpu().detach().numpy()
    if config.use_dynamic_gradient:
        e_queries = e_queries.argmax(-1)
        e_queries = e_queries.reshape(-1, 2)

    queries = e_queries.cpu().detach().numpy()

    plot_data = [
        # dict(data=queries, color="red", label="Previous e-queries"),
        dict(data=outputs, color="blue", label="New queries"),
    ]
    for i in plot_data:
        # rand from -0.5 to 0.5
        rand = np.random.rand(*i["data"].shape) - 0.5
        data = i["data"] + rand
        ax[0, 1].scatter(
            data[..., 0], data[..., 1],
            color=i["color"], s=1, alpha=0.5,
        )
    plt.savefig(name)
    plt.close()



def plot_posterior_samples(
    config,
    generator,
    embedder,
    reward_model,
    oracle,
    name,
    e_queries,
    outputs,
):
    r"""Plot the sample from 2D generator.

    When the design point is in 2D, we can perform
    evaluation by drawing the reward posterior. Did RL
    generate sample that is proportional to the posterior?
    """
    n_panels = config.n_samples + 1
    n_row = 1
    n_col = config.n_samples + 1
    _, ax = plt.subplots(
        int(n_row), int(n_col),
        squeeze=False, figsize=(20, 20)
    )

    # hidding axis ticks and label
    # and make the plot square
    for i in range(n_panels):
        row = i // n_col
        col = i % n_col 
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
        ax[row, col].set_xlabel("")
        ax[row, col].set_ylabel("")
        ax[row, col].set_aspect("equal")

    vocab_size_generator = oracle.vocab_size_generator
    ploting_points = 30
    levels = 50
    step_size = vocab_size_generator // ploting_points
    
    X = np.arange(0, vocab_size_generator+1, step=step_size)
    Y = np.arange(0, vocab_size_generator+1, step=step_size)
    ploting_points = X.shape[0]
    X[-1] = vocab_size_generator - 1
    Y[-1] = vocab_size_generator - 1
    X, Y = np.meshgrid(X, Y)
    XY = torch.tensor(np.array([X, Y])).to(e_queries.device)
    # >>> 2 x vocab_size_generator x vocab_size_generator

    if config.use_dynamic_gradient:
        # convert XY to one-hot
        XY = F.one_hot(XY, num_classes=vocab_size_generator).float()
        # >>> 2 x vocab_size_generator x vocab_size_generator 
        # ... x vocab_size_generator

    XY = embedder(sentence=XY, g2e_transl=generator.g2e_transl)
    # >>> 2 x vocab_size_generator x vocab_size_generator

    XY = XY.permute(1, 2, 0)
    # >>> vocab_size_generator x vocab_size_generator x 2

    XY = XY.reshape(-1, XY.shape[-1])

    Z_GT = oracle.scorer(XY)
    Z_GT = Z_GT.squeeze(-1).detach().cpu().numpy()
    Z_GT = Z_GT.reshape(ploting_points, ploting_points)

    Z = reward_model.posterior_function(XY)
    Z_mean = Z.mean(0).squeeze(-1).detach().cpu().numpy()
    Z_mean = Z_mean.reshape(ploting_points, ploting_points)
    ax[0, 0].contourf(X, Y, Z_mean, levels=levels, cmap="bwr", alpha=0.7)
    ax[0, 0].title.set_text("Posterior mean")

    for i in range(1, n_panels):
        row = i // n_col
        col = i % n_col
        Zi = Z[i-1].squeeze(-1).detach().cpu().numpy()
        Zi = Zi.reshape(ploting_points, ploting_points)
        ax[row, col].contourf(X, Y, Zi, levels=levels, cmap="bwr", alpha=0.7)
        ax[row, col].title.set_text(f"Posterior sample {i}")

    plt.savefig(name)
    plt.close()
