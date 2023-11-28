#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional, List

import torch
import gpytorch
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import (
    Kernel, RBFKernel, ScaleKernel, MaternKernel, CosineKernel
)
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)
from torch import Tensor
from botorch.sampling import SobolQMCNormalSampler
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import base_distributions
from gpytorch.likelihoods import Likelihood
from gpytorch.models import ExactGP


class VariationalPreferentialGP(GPyTorchModel, ApproximateGP):
    def __init__(
        self,
        queries: Tensor,
        responses: Tensor,
        bounds: List[int, int],
        use_withening: bool = True,
        covar_module: Optional[Kernel] = None,
    ) -> None:
        r"""
        Args:
            queries: A `n x q x d` tensor of training inputs. Each of the `n` queries is constituted
                by `q` `d`-dimensional decision vectors.
            responses: A `n x 1` tensor of training outputs. Each of the `n` responses is an integer
                between 0 and `q-1` indicating the decision vector selected by the user.
            use_withening: If true, use withening to enhance variational inference.
            covar_module: The module computing the covariance matrix.
        """
        train_x_induced = 1000
        if queries.shape[0] < train_x_induced:
            idx = torch.randperm(queries.shape[0])
        else:
            idx = torch.randperm(queries.shape[0])[:train_x_induced]

        queries = queries[idx]
        responses = responses[idx]

        self.queries = queries
        self.responses = responses
        self.input_dim = queries.shape[-1]
        self.q = queries.shape[-2]
        self.num_data = queries.shape[-3]

        # Reshape queries in the form of "standard training inputs"
        train_x = queries.reshape(
            queries.shape[0] * queries.shape[1], queries.shape[2]
        )

        # Squeeze out output dimension
        train_y = responses.squeeze(-1)
        
        # This assumes the input space has been normalized beforehand
        bounds = torch.tensor(
            [bounds for _ in range(self.input_dim)], dtype=torch.double
        ).T

        # Construct variational distribution and strategy
        if use_withening:
            inducing_points = draw_sobol_samples(
                bounds=bounds,
                n=2 * self.input_dim,
                q=1,
                seed=0,
            ).squeeze(1).to(train_x.device)
            inducing_points = torch.cat([inducing_points, train_x], dim=0)
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(-2)
            )
            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            )
        else:
            inducing_points = train_x
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(-2)
            )
            variational_strategy = UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            )

        super().__init__(variational_strategy)
        self.likelihood = PreferentialSoftmaxLikelihood(
            num_alternatives=self.q
        )
        self.mean_module = ConstantMean()
        scales = bounds[1, :] - bounds[0, :]

        if covar_module is None:
            self.covar_module = ScaleKernel(
                RBFKernel(
                    ard_num_dims=self.input_dim,
                    lengthscale_prior=GammaPrior(3.0, 6.0 / scales),
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
        else:
            self.covar_module = covar_module

        self._num_outputs = 1
        self.train_inputs = (train_x,)
        self.train_targets = train_y

    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_X = self.mean_module(X)
        covar_X = self.covar_module(X)
        return MultivariateNormal(mean_X, covar_X)

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1


class PreferentialSoftmaxLikelihood(Likelihood):
    r"""
    Implements the softmax likelihood used for GP-based preference learning.

    .. math::
        p(\mathbf y \mid \mathbf f) = \text{Softmax} \left( \mathbf f \right)

    :param int num_alternatives: Number of alternatives (i.e., q).
    """

    def __init__(self, num_alternatives):
        super().__init__()
        self.num_alternatives = num_alternatives
        self.noise = torch.tensor(1e-4)
        # This is only used to draw RFFs-based
        # samples. We set it close to zero because we want noise-free samples
        
        self.sampler = SobolQMCNormalSampler(sample_shape=512)
        # This allows for
        # SAA-based optimization of the ELBO

    def _draw_likelihood_samples(
        self, function_dist, *args, sample_shape=None, **kwargs
    ):
        function_samples = self.sampler(GPyTorchPosterior(function_dist)).squeeze(-1)
        return self.forward(function_samples, *args, **kwargs)

    def forward(self, function_samples, *params, **kwargs):
        function_samples = function_samples.reshape(
            function_samples.shape[:-1]
            + torch.Size(
                (
                    int(function_samples.shape[-1] / self.num_alternatives),
                    self.num_alternatives,
                )
            )
        )  # Reshape samples as if they came from a multi-output model (with `q` outputs)
        num_alternatives = function_samples.shape[-1]

        if num_alternatives != self.num_alternatives:
            raise RuntimeError("There should be %d points" % self.num_alternatives)

        res = base_distributions.Categorical(logits=function_samples)  # Passing the
        # function values as logits recovers the softmax likelihood
        return res


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


class VariationalDirichletGPModel(GPyTorchModel, ApproximateGP):
    def __init__(
        self,
        queries: Tensor,
        responses: Tensor,
        bounds: List[int, int],
        num_classes: int,
        likelihood: Likelihood = None,
        use_withening: bool = True,
    ) -> None:
        self.queries = queries
        self.responses = responses
        self.input_dim = queries.shape[-1]
        self.num_data = queries.shape[-2]

        # Reshape queries in the form of "standard training inputs"
        train_x = queries

        # Squeeze out output dimension
        train_y = responses.squeeze(-1)
        
        # This assumes the input space has been normalized beforehand
        bounds = torch.tensor(
            [bounds for _ in range(self.input_dim)], dtype=torch.double
        ).T
        
        train_x_induced = 1000
        if train_x.shape[0] < train_x_induced:
            idx = torch.randperm(train_x.shape[0])
        else:
            idx = torch.randperm(train_x.shape[0])[:train_x_induced]

        # Construct variational distribution and strategy
        if use_withening:
            inducing_points = draw_sobol_samples(
                bounds=bounds,
                n=self.input_dim,
                q=1,
                seed=0,
            ).squeeze(1).to(train_x)
            inducing_points = torch.cat([inducing_points, train_x[idx]], dim=0)
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(-2)
            )
            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            )
        else:
            inducing_points = train_x[idx]
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(-2)
            )
            variational_strategy = UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
            )
        super().__init__(variational_strategy)
        self.likelihood = likelihood
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))

        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )
        
        self.train_inputs = (train_x,)
        self.train_targets = train_y
        self.temperature = torch.nn.Parameter(torch.ones(num_classes))
            
    def forward(self, X: Tensor) -> MultivariateNormal:
        mean_X = self.mean_module(X)
        covar_X = self.covar_module(X)
        return MultivariateNormal(mean_X, covar_X)
    
    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1
    
    def platt_scale(self, logits):
        """
        Perform Platt's scaling on logits
        
        Args:
            logits: n_data x num_classes
        """
        # Expand temperature to match the size of logits
        original_shape = logits.shape
        logits = logits.reshape(-1, logits.shape[-1])
        temperature = self.temperature[None, ...].expand_as(logits)
        return (logits / temperature).reshape(original_shape)

    def rescale_probs(self, probs):
        """
        Perform Platt's scaling on probs.

        Args:
            probs: n_data x num_classes
    
        Returns:
            torch.Tensor: Tensor shape `n_data x num_classes` of scaled logits
        """

        probs = probs * torch.relu(self.temperature[None, ...])
        return probs / probs.sum(dim=-1, keepdim=True)