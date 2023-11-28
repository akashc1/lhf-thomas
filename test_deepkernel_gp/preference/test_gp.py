import torch
from my_utils.gp import VariationalPreferentialGP as VPGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.variational_elbo import VariationalELBO
import matplotlib.pyplot as plt
from botorch.test_functions import Ackley
from botorch.models.pairwise_gp import (
    PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
)
from botorch.models.transforms.input import Normalize
from scipy.stats import kendalltau
from tqdm import tqdm
from gpytorch.kernels import (
    ScaleKernel, MaternKernel
)
from gpytorch.priors.torch_priors import GammaPrior

# set seed for reproducibility
torch.manual_seed(0)

bounds = [0, 1]
scale = bounds[1] - bounds[0]
n_dim = 1
f = Ackley(dim=n_dim).double()
f.bounds[0, :].fill_(bounds[0])
f.bounds[1, :].fill_(bounds[1])

ktc_metrics = dict(
    pairwiseGP = [],
    VPGP = [],
)
exp_range = range(2, 500, 2)
for n_comparison in tqdm(exp_range):
    train_X = torch.rand(2*n_comparison, 1).double()
    train_Y_ = f(train_X).double()
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    # plot data
    ax.set_title("GP model")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, 1)

    # ground truth function
    x = torch.linspace(0, 1, 100).unsqueeze(-1)
    y = f(x)
    ax.plot(x.squeeze(-1), y, label="f(x)")
    ax.scatter(train_X.squeeze(-1), train_Y_, label="Data")

    for i, model in enumerate(["pairwiseGP", "VPGP"]):
        if model == "pairwiseGP":
            train_Y = []
            while len(train_Y) < n_comparison:
                # select 2 random index in range [0, 40) that are not the same
                idx = torch.randint(0, 2*n_comparison, (2,))
                while idx[0] == idx[1]:
                    idx = torch.randint(0, 2*n_comparison, (2,))
                
                # select the one with the highest value
                if train_Y_[idx[0]] == train_Y_[idx[1]]:
                    continue
                elif train_Y_[idx[0]] > train_Y_[idx[1]]:
                    train_Y.append([idx[0], idx[1]])
                else:
                    train_Y.append([idx[1], idx[0]])

            train_Y = torch.tensor(train_Y).double()

            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=n_dim,
                    lengthscale_prior=GammaPrior(3.0, 6.0 / scale),
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            reward_model = PairwiseGP(
                train_X, train_Y, 
                input_transform=Normalize(d=train_X.shape[-1]),
                covar_module=covar_module,
            ).double()
            mll = PairwiseLaplaceMarginalLogLikelihood(
                reward_model.likelihood, reward_model
            )
            mll = fit_gpytorch_mll(mll)
        elif model == "VPGP":
            train_X = train_X.reshape(-1, 2, 1)
            train_Y_ = f(train_X).double()
            train_Y = train_Y_.argmax(dim=-1, keepdim=True)

            reward_model = VPGP(train_X, train_Y, bounds=[0, 1]).double()

            mll = VariationalELBO(
                likelihood=reward_model.likelihood,
                model=reward_model,
                num_data=2 * reward_model.num_data,
            )
            mll = fit_gpytorch_mll(mll)

            train_X = train_X.reshape(-1, 1)

        # plot posterior
        posterior = reward_model.posterior(x)
        std= posterior.variance.sqrt().squeeze(-1).detach().numpy()
        mean = posterior.mean.squeeze(-1).detach().numpy()
        color = "red" if model == "pairwiseGP" else "blue"
        ax.plot(x.squeeze(-1), mean, label=f"PM {model}", color=color)
        ax.fill_between(x.squeeze(-1), mean-std/4, mean+std/4, alpha=0.25, color=color)

        ktc = kendalltau(mean, y).correlation
        print(f"Kendall-Tau rank correlation for {model}: {ktc:.4f}")

        ktc_metrics[model].append(ktc)

    ax.legend(loc="upper right")
    plt.savefig(f"gp_{n_comparison}.pdf")
    plt.close()

fig, ax = plt.subplots(1, 1)
ax.set_title("Kendall-Tau rank correlation")
ax.set_xlabel("Number of comparisons")
ax.set_ylabel("Kendall-Tau rank correlation")
ax.set_xticks(list(exp_range))
ax.set_xticklabels(list(exp_range))
ax.plot(list(exp_range), ktc_metrics["pairwiseGP"], label="pairwiseGP")
ax.plot(list(exp_range), ktc_metrics["VPGP"], label="VPGP")

ax.legend(loc="upper right")
plt.savefig("gp_ktc.pdf")
plt.close()