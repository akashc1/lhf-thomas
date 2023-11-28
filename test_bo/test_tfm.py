import torch
from tqdm import tqdm
from torch.distributions import Categorical
from my_utils.tfm import TFM
import matplotlib.pyplot as plt
import torch.nn.functional as F

# =========================
# ==== Test TFM 1-Step ====
# =========================

# batch_size = 100
# prompt_length = 1
# vocab_size_generator = 20
# iter = 1000
# theta = TFM(
#     use_SAA=False,
#     max_batch_size=1,
#     max_length=1+1,
#     max_n_restart=10,
#     max_q=1,
#     bounds_embedder=[-1, 1],
#     vocab_size_generator=20,
#     d_model=4,
#     nhead=1,
#     num_layers=2,
#     dim_feedforward=32,
# ).to("cuda")

# optim = torch.optim.Adam(theta.parameters(), lr=1e-3)
# sm = torch.nn.Softmax(dim=-1)
# diffs = []
# for i in tqdm(range(iter)):
#     x = torch.randint(0, 20, (batch_size, prompt_length)).cuda()
#     # (batch_size, prompt_length)

#     x_onehot = torch.nn.functional.one_hot(
#         x, num_classes=vocab_size_generator
#     ).float()
#     # (batch_size, prompt_length, vocab_size)
    
#     x_onehot = x_onehot.permute(1, 0, 2)
#     # (prompt_length, batch_size, vocab_size)

#     next_x_logit = theta(x_onehot)[-1, :, :]
#     # >>> batch_size, vocab_size

#     probs = sm(next_x_logit)
#     # (batch_size, vocab_size)

#     next_x_dist = Categorical(probs=probs)
    
#     mnll = -next_x_dist.log_prob(x.squeeze(-1)).mean()
    
#     mnll.backward()
#     optim.step()
#     optim.zero_grad()

#     pred = probs.argmax(dim=-1)
#     diff = abs(pred - x.squeeze(-1)).float().mean()
#     diffs.append(diff.item())

#     if i % 100 == 0:
#         print("i:", i, "mnll:", mnll.item(), "diff:", diff.item())

# plt.figure()
# plt.plot(diffs)
# plt.savefig("loss.pdf")
# plt.close()

# =========================
# ==== Test TFM 3-Step ====
# =========================

batch_size = 100
prompt_length = 1
vocab_size_generator = 20 # 60000
iter = 10000
max_length = 10 #1000
lr = 1e-3
theta = TFM(
    use_SAA=False,
    max_batch_size=batch_size+1,
    max_length=max_length,
    max_n_restart=10,
    max_q=1,
    bounds_embedder=[-1, 1],
    vocab_size_generator=vocab_size_generator,
    d_model=32,
    nhead=1,
    num_layers=2,
    dim_feedforward=32,
).to("cuda")

optim = torch.optim.Adam(theta.parameters(), lr=lr)
sm = torch.nn.Softmax(dim=-1)
losses = []
vec = torch.arange(vocab_size_generator).cuda()

for i in tqdm(range(iter)):
    x = torch.randint(0, vocab_size_generator, (batch_size,1)).cuda()
    # >>> (batch_size, prompt_length)

    y = x + torch.arange(max_length-1)[None].cuda()
    # >>> (batch_size, max_length - prompt_length)

    y = y % vocab_size_generator

    x_onehot = F.one_hot(
        x, num_classes=vocab_size_generator
    ).float()
    # >>> (batch_size, max_length - prompt_length, vocab_size)

    x_onehot = x_onehot.permute(1, 0, 2)
    # >>> (prompt_length, batch_size, vocab_size)

    _, outputs_probs, *_ = theta.generate(
        x_onehot, n_restart=1, q=1, max_length=max_length
    )
    # >>> (batch_size x n_restart x q x max_length x vocab_size)

    outputs_probs = outputs_probs[:, 0, 0, 1:, :]
    # >>> (batch_size, max_length - prompt_length, vocab_size)

    dist = Categorical(probs=outputs_probs)
    loss = -dist.log_prob(y).mean()
    loss.backward()
    optim.step()
    optim.zero_grad()
    losses.append(loss.item())

    if i % 100 == 0:
        print("i:", i, "loss:", loss.item())

# plot loss 
plt.figure()
plt.plot(losses)
plt.savefig("loss.pdf")
plt.close()
