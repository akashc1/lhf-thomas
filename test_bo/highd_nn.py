import torch
from botorch.test_functions import Ackley
from scipy.stats import kendalltau
from my_utils.dense import DenseModel
from my_utils.synthfunc import SynGP

n_dim = 768
seed = 0
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
n_comparison = n_dim * 30
bounds = [-1, 1]
q = 2
n_epochs = 10000
lr = 1e-3
n_samples = 100
reward_network_info = dict(
    layers=4,
    node_size=1024,
    activation_function=torch.nn.ELU(),
    last_activation=None,
    dropout_rate=0.2,
)

# Generate data
# f = Ackley(dim=n_dim).to(device, dtype)
f = SynGP(dim=n_dim).to(device, dtype)
f.bounds[0, :].fill_(bounds[0])
f.bounds[1, :].fill_(bounds[1])

train_X = torch.rand(n_comparison, q, n_dim).to(device, dtype)
# train_X = train_X / train_X.norm(dim=-1, keepdim=True)
# >>> n_comparison x q x n_dim

train_Y_ = f(train_X)
# >>> n_comparison x q

train_Y = f(train_X).argmax(dim=-1)
# >>> n_comparison

# Train reward model
reward_model = DenseModel(
    input_size=n_dim,
    output_shape=1,
    info=reward_network_info,
).to(device, dtype)
optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)

for i in range(n_epochs):
    output_probs = reward_model(train_X).squeeze(-1)
    # >>> n_comparison x q

    dist = torch.distributions.Categorical(probs=output_probs)
    loss = -dist.log_prob(train_Y).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % (n_epochs//50) == 0:
        print(f"Iter {i+1}/{n_epochs} - Loss: {loss.item():.3f}")

# Evaluation using Kendall Tau rank correlation
x = torch.rand(n_comparison, q, n_dim).to(device, dtype)
# x = x / x.norm(dim=-1, keepdim=True)
y = f(x).cpu().detach().numpy()

with torch.no_grad():
    mean = [reward_model(x).squeeze(-1) for _ in range(n_samples)]
    mean = torch.stack(mean, dim=0).mean(dim=0)

mean = mean.cpu().detach().numpy()
ktc = kendalltau(mean, y).correlation
print(f"Kendall Tau rank correlation for {n_dim} dimension: {ktc:.4f}")

mean_prediction = mean.argmax(axis=-1)
y_prediction = y.argmax(axis=-1)
accuracy = (mean_prediction == y_prediction).mean()
print(f"Accuracy: {accuracy:.4f}")
