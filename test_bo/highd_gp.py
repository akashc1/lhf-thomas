import torch
from gpytorch.mlls.variational_elbo import VariationalELBO
from botorch.test_functions import Ackley
from scipy.stats import kendalltau
from my_utils.gp import VariationalPreferentialGP as VPGP


n_dim = 2
seed = 0
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
n_comparison = n_dim * 40
bounds = [-1, 1]
q = 2
n_epochs = 100
batch_size = 100
lr = 1e-4
n_samples = 100

# Generate data
f = Ackley(dim=n_dim).to(device, dtype)
f.bounds[0, :].fill_(bounds[0])
f.bounds[1, :].fill_(bounds[1])

train_X = torch.randn(n_comparison, q, n_dim).to(device, dtype)
train_X = train_X / train_X.norm(dim=-1, keepdim=True)
# >>> n_comparison x q x n_dim

train_Y_ = f(train_X)
# >>> n_comparison x q

train_Y = f(train_X).argmax(dim=-1, keepdim=True)
# >>> n_comparison x 1

# Train reward model
reward_model = VPGP(train_X, train_Y, bounds)
reward_model = reward_model.to(device).to(dtype)
mll = VariationalELBO(
    likelihood=reward_model.likelihood,
    model=reward_model,
    num_data=2 * reward_model.num_data,
)
optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)

for i in range(n_epochs):
    idx = torch.randperm(n_comparison)[:batch_size]
    output = reward_model(train_X[idx])
    loss = -mll(output, train_Y[idx]).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Iter {i+1}/{batch_size} - Loss: {loss.item():.3f}")

# Evaluation using Kendall Tau rank correlation
x = torch.rand(n_comparison, n_dim).to(device)
x = x / x.norm(dim=-1, keepdim=True)
y = f(x).cpu().detach().numpy()
posterior = reward_model.posterior(x)
samples = posterior.sample(sample_shape=torch.Size([n_samples]))
mean = posterior.mean.squeeze(-1).cpu().detach().numpy()
std = posterior.variance.sqrt().squeeze(-1).cpu().detach().numpy()
ktc = kendalltau(mean, y).correlation
print(f"Kendall Tau rank correlation for {n_dim} dimension: {ktc:.4f}")
