import torch
from my_utils.cat_with_base import OneHotCategoricalSTVarianceReduction as OHCTVR
import matplotlib.pyplot as plt
from tqdm import tqdm


def reward(x, d):
    p_r1 = 0.6
    p_r2 = 0.3
    p_r3 = 0.1

    r1 = torch.zeros(d).cuda()
    r1[0] = p_r1/p_r1
    r2 = torch.zeros(d).cuda()
    r2[-1] = 0.95 * p_r1/p_r2
    r3 = torch.zeros(d).cuda()
    r3[-2] = 0.95 * p_r1/p_r3

    flip = torch.rand(1)
    if flip < p_r1:
        r = r1
    elif flip < p_r1 + p_r2:
        r = r2
    else:
        r = r3

    return (x * r).sum(-1)

d = 10
t = 0.1
iter = 5000
lr = 0.001
n_samples = 1000
w_size = 100
device = "cuda"

theta = torch.rand(d, requires_grad=True, device=device)
optimizer = torch.optim.Adam([theta], lr=lr)
sm = torch.nn.Softmax(dim=-1)
losses = []

for i in tqdm(range(iter)):
    dist = OHCTVR(probs=sm(theta/t))
    x = dist.rsample((n_samples,))
    loss = -reward(x, d=d).mean() - t * dist.entropy()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    losses.append(loss)
    if i%100 == 0:
        _theta = sm(theta).cpu().detach().numpy().round(2)
        print(f"theta = {_theta}")

losses = torch.tensor(losses)
smoothed_losses = torch.nn.functional.conv1d(
    losses.view(1, 1, -1),
    torch.ones(1, 1, w_size)/w_size, padding=5
).view(-1).numpy()
losses = losses.cpu().detach().numpy()
plt.plot(losses, label='loss', alpha=0.5)
plt.plot(smoothed_losses, label='smoothed loss')
_theta = sm(theta).cpu().detach().numpy().round(2)
plt.title(f'theta = {_theta}') if d <= 10 else print(_theta)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.savefig('test_sto_optim.pdf')
plt.close()
