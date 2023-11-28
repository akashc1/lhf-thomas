import torch
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from gpytorch.kernels import RBFKernel, ScaleKernel
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(0)

# Generate synthetic data
train_X = torch.rand(20, 1)
train_Y = torch.sin(4 * train_X) + 0.1 * torch.randn_like(train_X)
true_Y = torch.sin(4 * train_X)

# Generate pairwise preferences
train_Y_idx = torch.argsort(train_Y.squeeze(), descending=True)
train_Y_pref = torch.stack([train_Y_idx[:-1], train_Y_idx[1:]]).T

# Initialize the PairwiseGP model
covar_module = ScaleKernel(RBFKernel())
gp_model = PairwiseGP(train_X, train_Y_pref, covar_module=covar_module).double()
mll = PairwiseLaplaceMarginalLogLikelihood(gp_model.likelihood, gp_model)

# Fit the model
fit_gpytorch_model(mll)

# Evaluate the model
test_X = torch.linspace(0, 1, 100).view(-1, 1)
posterior = gp_model.posterior(test_X)
mean = posterior.mean.squeeze().detach().numpy()
std = posterior.variance.sqrt().squeeze().detach().numpy()

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(train_X, true_Y, 'kx', label='True data')
plt.plot(test_X, mean, 'b-', label='Posterior mean')
plt.fill_between(test_X.squeeze(), mean - std, mean + std, color='b', alpha=0.2, label='Posterior std. dev.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('PairwiseGP model with synthetic data')
plt.legend()
plt.show()
