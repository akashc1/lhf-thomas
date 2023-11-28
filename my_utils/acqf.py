from __future__ import annotations

from typing import Optional

import copy
import torch
from torch import Tensor
from botorch.acquisition import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)


class qExpectedUtilityOfBestOption(MCAcquisitionFunction):
    r"""Expected Utility of Best Option (qEUBO).

    This computes qEUBO by
    (1) sampling the joint posterior over q points
    (2) evaluating the maximum objective value accross the q points for each sample
    (3) averaging over the samples

    `qEUBO(X) = E[max Y], Y ~ f(X), where X = (x_1,...,x_q)`
    """

    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        n_fantasy: Optional[int] = 64,
        objective: Optional[MCAcquisitionObjective] = None,
        X_baseline: Optional[Tensor] = None,
    ) -> None:
        r"""MC-based Expected Utility of the Best Option (qEUBO).

        Args:
            model: A fitted model.
             sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            X_baseline:  A `m x d`-dim Tensor of `m` design points forced to be included
                in the query (in addition to the q points, so the query is constituted
                by q + m alternatives). Concatenated into X upon forward call. Copied and
                set to have no gradient. This is useful, for example, if we want to force
                one of the alternatives to be the point chosen by the decision-maker in
                the previous iteration.
        """

        if sampler is None:
            if n_fantasy is None:
                raise ValueError(
                    "Must specify `n_fantasy` if no `sampler` is provided."
                )
            # base samples should be fixed for joint optimization
            sampler = SobolQMCNormalSampler(
                sample_shape=n_fantasy,
                resample=False,
                collapse_batch_dims=True,
            )
        elif n_fantasy is not None:
            if sampler.sample_shape != torch.Size([n_fantasy]):
                raise ValueError("The sampler shape must match {n_fantasy}.")
        else:
            n_fantasy = sampler.sample_shape[0]

        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            X_pending=X_baseline,
        )

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qEUBO on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of qEUBO values at the
            given design points `X`, where `batch_shape'` is the broadcasted batch shape
            of model and input `X`.
        """
        model = copy.deepcopy(self.model)
        posterior_X = model.posterior(X)
        Y_samples = self.sampler(posterior_X)
        # >>> n_samples x batch_shape x q x 1

        Y_samples = Y_samples.squeeze(-1)
        # >>> n_samples x batch_shape x q

        Y_best = Y_samples.max(dim=-1).values
        # >>> n_samples x batch_shape

        return Y_best.mean(dim=0)


if __name__ == "__main__":
    # import
    from botorch.models import SingleTaskGP
    import matplotlib.pyplot as plt
    from botorch.test_functions import Ackley
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_model

    # set random seed
    torch.manual_seed(0)

    # create random data in range(0,1)
    train_X = torch.rand(5, 1).double()

    # define oracle function to be Ackley function import from Botorch
    f = Ackley(dim=1)
    f.bounds[0, :].fill_(0)
    f.bounds[1, :].fill_(1)
    train_Y = f(train_X).unsqueeze(-1)

    # define singletaskGP model
    model = SingleTaskGP(train_X, train_Y).double()

    # fit model
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # define qEUBO acquisition function
    qeubo = qExpectedUtilityOfBestOption(model=model, n_fantasy=64)

    # define candidate set
    X = torch.rand(2, 1)

    # save a copy of unoptimized X
    X_unoptimized = X.clone()

    # set X to be optimized
    X.requires_grad = True

    # perform joint optimization with torch
    optim = torch.optim.Adam([X], lr=0.01)
    losses = []

    for i in range(5000):
        loss = -qeubo(torch.sigmoid(X))
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss.item())

    X = torch.sigmoid(X)

    # plotting loss curve
    fig = plt.figure(figsize=(10, 10))
    plt.plot(losses, label="loss curve")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()

    # plotting ground truth function
    fig = plt.figure(figsize=(10, 10))

    x = torch.linspace(0, 1, 1000).double().unsqueeze(-1)
    y = f(x)

    # plot posterior
    posterior = model.posterior(x)
    mean = posterior.mean.detach().numpy().squeeze(-1)
    std = posterior.variance.sqrt().detach().numpy().squeeze(-1)

    x = x.squeeze(-1).detach().numpy()
    y = y.squeeze(-1).detach().numpy()
    plt.plot(x, y, label="ground truth function")
    plt.plot(x, mean, label="posterior mean")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2, label="posterior std")

    # plot training points
    plt.scatter(train_X, train_Y, label="training points")

    # plot new candidate points
    plt.scatter(
        X.cpu().detach().numpy(),
        f(X).cpu().detach().numpy(),
        label="new candidate points",
    )

    # plot unoptimized candidate points
    plt.scatter(
        X_unoptimized.cpu().detach().numpy(),
        f(X_unoptimized).cpu().detach().numpy(),
        label="unoptimized candidate points",
    )

    # save plot
    plt.legend()
    plt.savefig("plot.png")
    plt.close()
