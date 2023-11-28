import math
import torch
import numpy as np
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.models import ExactGP
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from my_utils.gp import VariationalDirichletGPModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_data = 8
bounds = [-4, 4]

def gen_data(num_data, device, seed = 2023):
    torch.random.manual_seed(seed)
    
    x = torch.randn(num_data,1).to(device)
    u = torch.ones(1).to(x.device)
    data_fn = lambda x: torch.sin(0.15 * u * 3.1415 * x * 2) + 1
    latent_fn = data_fn(x)
    z = torch.round(latent_fn).long().squeeze()
    return x, z, data_fn

train_x, train_y, genfn = gen_data(num_data=num_data, device=device)

# We will use the simplest form of GP model, exact inference
class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# breakpoint()
# initialize likelihood and model
# we let the DirichletClassificationLikelihood compute the targets for us
likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True).to(device)
# model = DirichletGPModel(train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes).to(device)
model = VariationalDirichletGPModel(
    queries=train_x,
    responses=likelihood.transformed_targets,
    bounds=bounds,
    num_classes=likelihood.num_classes,
    likelihood=likelihood,
    # use_withening=False
).to(device)

# this is for running the notebook in our testing framework
training_iter = 500

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
# mll = VariationalELBO(
mll = PredictiveLogLikelihood(
    likelihood=likelihood,
    model=model,
    num_data=model.num_data,
)

mll = mll.to(device)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, likelihood.transformed_targets).sum()
    loss.backward()
    if i % 5 == 0:
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.mean().item(),
            model.likelihood.second_noise_covar.noise.mean().item()
        ))
    optimizer.step()
    
    
model.eval()
likelihood.eval()

test_x = torch.linspace(bounds[0], bounds[1], 200).to(device)
test_labels = torch.round(genfn(test_x)).long().squeeze().to(device)

with gpytorch.settings.fast_pred_var(), torch.no_grad():
    test_dist = model(test_x)
    
    pred_means = test_dist.loc
    pred_stds = test_dist.stddev
    
    probs = test_dist.sample(torch.Size((256,)))
    # >>> 256 x num_classes x num_test
    probs = probs.softmax(dim=1)
    # >>> 256 x num_classes x num_test
    probs_means = probs.mean(dim=0)
    probs_stds = probs.std(dim=0)
# breakpoint()
    
# Plot the results
# The plot must have a ground truth sine wave,
# a black line for the predicted mean, and a shaded std area
# around the mean.
# The plot must also have the training data as black stars.
# YOUR CODE HERE
for c in range(likelihood.num_classes):
    plt.figure(figsize=(8, 6))
    plt.plot(test_x.cpu().numpy(), probs_means[c].cpu().numpy(), 'k', label='Predicted probability')
    plt.fill_between(test_x.cpu().numpy(), 
                     probs_means[c].cpu().numpy() - probs_stds[c].cpu().numpy(), 
                     probs_means[c].cpu().numpy() + probs_stds[c].cpu().numpy(), alpha=0.5, label='Probability confidence')
    # plt.plot(test_x.cpu().numpy(), pred_means[c].cpu().numpy(), 'k', label='Predicted mean')
    # plt.fill_between(test_x.cpu().numpy(), 
    #                  pred_means[c].cpu().numpy() - pred_stds[c].cpu().numpy(), 
    #                  pred_means[c].cpu().numpy() + pred_stds[c].cpu().numpy(), alpha=0.5, label='Confidence')
    
    plt.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k*', label='Training samples') # training data
    plt.plot(test_x.cpu().numpy(), test_labels.cpu().numpy(), label='Groundtruth class') # ground truth
    # plt.plot(test_x.cpu().numpy(), genfn(test_x).cpu().numpy())
    
    plt.ylim(-1, 3)
    plt.legend()
    plt.title('Predicted mean and confidence interval')
    plt.savefig(f"test_{c}.pdf")
