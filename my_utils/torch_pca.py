import torch
from torch import Tensor


class PCA:
    def __init__(
        self, n_components: int, niter: int = 1000
    ) -> None:
        self.n_components = n_components
        self.niter = niter

    def fit(self, X, y=None) -> None:
        r"""Perform principal component analysis on the input data.
        
        The output is U, S, V such that X = U @ S @ V.T.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
        """
        U, S, V = torch.pca_lowrank(
            X, q = self.n_components, niter=self.niter
        )
        self.V = V

    def transform(self, X: Tensor) -> Tensor:
        r"""Project the input data onto the principal components.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Projected data of shape (n_samples, n_components).
        """
        return X @ self.V

if __name__ == "__main__":
    pca = PCA(n_components=2)

    # generate 2D normally distributed data
    X = torch.randn(1000, 2) + 5
    X = X.to("cuda")

    # perform PCA
    pca.fit(X)

    # project data onto principal components
    X_pca = pca.transform(X)

    # save with torch.save
    torch.save(pca, "pca.pt")

    # load with torch.load
    pca = torch.load("pca.pt")


    # # compare with sklearn
    # from sklearn.decomposition import PCA as PCA_sk
    # pca_sk = PCA_sk(n_components=2)
    # pca_sk.fit(X)
    # X_pca_sk = pca_sk.transform(X)

    # # plot results
    # import matplotlib.pyplot as plt
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], label="torch", alpha=0.5)
    # plt.scatter(X_pca_sk[:, 0], X_pca_sk[:, 1], label="sklearn", alpha=0.5)
    # plt.legend()
    # plt.savefig("pca.pdf")
