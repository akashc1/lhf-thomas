import torch.nn as nn
from torch import Tensor
from my_utils.decorator import FreezeParameters


class qExpectedUtilityOfBestOption(nn.Module):
    def forward(self, model:nn.Module, X: Tensor) -> Tensor:
        r"""Evaluate qEUBO on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches 
                with `q` `d`-dim design points each.
            model: A nn.Module model that support posterior computation.
        Returns:
            A `batch_shape'`-dim Tensor of qEUBO values at the
            given design points `X`, where `batch_shape'` is the
            broadcasted batch shape of model and input `X`.
        """
        with FreezeParameters([model]):
            Y_samples = model.posterior(X)
            # >>> n_samples x batch_shape x q x 1

            Y_samples = Y_samples.squeeze(-1)
            # >>> n_samples x batch_shape x q

            Y_best = Y_samples.max(dim=-1).values
            # >>> n_samples x batch_shape

        return Y_best.mean(dim=0)

