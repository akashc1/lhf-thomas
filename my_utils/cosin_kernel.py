import torch
from gpytorch.kernels import Kernel

class CosineKernel(Kernel):
    def forward(self, x1, x2, **params):
        # Compute the kernel matrix as the dot product 
        # between x1 and x2
        return x1.matmul(x2.transpose(-1, -2))
    
if __name__ == "__main__":

    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.kernels import ScaleKernel
    from botorch.fit import fit_gpytorch_model

    train_X = torch.randn(10, 5)
    train_X = train_X / train_X.norm(dim=-1, keepdim=True)

    train_Y = torch.randn(10, 1)

    Kernel = ScaleKernel(CosineKernel())
    out = Kernel.forward(train_X, train_X)
    breakpoint()

    model = SingleTaskGP(
        train_X, 
        train_Y, 
        covar_module=ScaleKernel(CosineKernel())
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)