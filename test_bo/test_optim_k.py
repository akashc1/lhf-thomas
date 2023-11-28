import torch
from my_utils.cat_with_base import OneHotCategoricalSTVarianceReduction as OHCTVR
import matplotlib.pyplot as plt
from tqdm import tqdm


d = 20 # vocab size of generator
k = 3 # number of tokens
t = 0.1 # temperature
iter = 10000
lr = 1e-1
n_samples = 10 # n_samples at each decoding step
device = "cuda"

theta = torch.rand(*[d]*k, requires_grad=True, device=device)
optimizer = torch.optim.Adam([theta], lr=lr)
sm = torch.nn.Softmax(dim=-1)
vec = torch.arange(d, device=device)
losses = []

for i in tqdm(range(iter)):
    dist = OHCTVR(probs=sm(theta))
    outputs = dist.rsample((n_samples,))
    # >>> n_samples x [d]*k

    outputs = (outputs * vec).sum(-1)
    # >>> n_samples x [d]*(k-1)

    loss = abs(outputs - vec).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses.append(loss.item())
    if i % 1000 == 0:
        print(f"Acqf loss {i}/{iter}: {loss.item():.3f}")

# draw a heatmap of theta with colorbar
if k == 2:
    plt.figure()
    plt.imshow(sm(theta).cpu().detach().numpy())
    plt.colorbar()
    plt.savefig('test_sto_k_optim.pdf')
    plt.close()

plt.figure()
plt.plot(losses)
plt.savefig(f"test_sto_k_optim_loss.pdf")
plt.close()
