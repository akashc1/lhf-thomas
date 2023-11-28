r"""Variance Reduction using fixed base for categorical distribution."""

import torch
from torch.distributions import OneHotCategoricalStraightThrough
from torch import Tensor


class OneHotCategoricalSTVarianceReduction(OneHotCategoricalStraightThrough):
    r"""Variance Reduction for categorical distribution using base samples."""

    def rsample_with_base_sample(self, base_sample: Tensor):
        r"""Reparameterized sample with base samples.

        Args:
            base_sample (torch.Tensor): Base samples to use for variance
                reduction. Shape: batch_size
        Returns:
            torch.Tensor: Samples from the distribution.
                Shape: num_return_sequences x q x batch_size x num_categories
        """
        probs = self._categorical.probs  # cached via @lazy_property
        cdf = torch.cumsum(probs, dim=-1)
        # >>> *batch_size x num_categories

        base_sample = base_sample[..., None].to(cdf)
        # >>> *batch_size x 1
        
        breakpoint()
        
        indices = torch.searchsorted(cdf, base_sample).squeeze(-1)
        # >>> *batch_size

        num_events = self._categorical._num_events
        samples = torch.nn.functional.one_hot(indices, num_events).to(probs)
        # >>> *batch_size x num_categories

        samples = samples + (probs - probs.detach())

        return samples

    def rsample_mode(self):
        r"""Reparameterized sample with mode."""
        probs = self._categorical.probs

        indices = probs.argmax(dim=-1)
        num_events = self._categorical._num_events
        samples = torch.nn.functional.one_hot(indices, num_events).to(probs)
        # >>> *batch_size x num_categories

        return samples + (probs - probs.detach())

if __name__ == "__main__":
    prob = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    prob.requires_grad = True
    dist = OneHotCategoricalSTVarianceReduction(prob)
    
    samples = []
    for _ in range(1000):
        base_sample = torch.rand(1)
        sample = dist.rsample_with_base_sample(base_sample)
        # >>> still differentiable until here wrt prob
        samples.append(sample)
    samples = torch.stack(samples)
    samples = samples.argmax(dim=-1)

    # plot histogram
    import matplotlib.pyplot as plt

    plt.hist(samples.numpy(), bins=4, density=True)
    plt.savefig("cat_with_base.pdf")
    plt.close()
