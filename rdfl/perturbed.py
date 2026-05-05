from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Callable

import numpy as np
import torch
from torch import nn


Oracle = Callable[[torch.Tensor], torch.Tensor]


class _PerturbedOptimizerFunction(torch.autograd.Function):
    """Monte Carlo differentiable perturbed optimizer.

    Berthet et al. define the smoothed optimizer as

        y_sigma(theta) = E_Z argmax_y <theta + sigma Z, y>.

    For Gaussian Z, integration by parts gives

        J_theta y_sigma(theta) = E[y*(theta + sigma Z) Z^T] / sigma.

    This custom autograd function implements the corresponding vector-Jacobian
    product:

        J^T g = E[Z <y*, g>] / sigma.
    """

    @staticmethod
    def forward(
        ctx,
        theta: torch.Tensor,
        noise: torch.Tensor,
        sigma: float,
        antithetic: bool,
        oracle: Oracle,
    ) -> torch.Tensor:
        batch, dim = theta.shape
        if antithetic:
            noise = torch.cat([noise, -noise], dim=0)

        samples = []
        with torch.no_grad():
            for z in noise:
                perturbed = theta + sigma * z.unsqueeze(0)
                samples.append(oracle(perturbed))
            solutions = torch.stack(samples, dim=0)
            output = solutions.mean(dim=0)

        ctx.save_for_backward(noise, solutions)
        ctx.sigma = sigma
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        noise, solutions = ctx.saved_tensors
        sigma = ctx.sigma
        inner = (solutions * grad_output.unsqueeze(0)).sum(dim=-1)
        grad_theta = (noise.unsqueeze(1) * inner.unsqueeze(-1)).mean(dim=0) / sigma
        return grad_theta, None, None, None, None


class PerturbedOptimizerLayer(nn.Module):
    """Wraps a black-box discrete argmax oracle as a differentiable layer.

    The wrapped oracle receives score vectors `theta` and returns discrete
    solutions with the same shape. The forward output is the Monte Carlo
    average of perturbed discrete solutions, so it is continuous and can be
    used in downstream losses.
    """

    def __init__(
        self,
        oracle: Oracle,
        decision_dim: int,
        num_samples: int = 16,
        sigma: float = 1.0,
        antithetic: bool = True,
    ) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.oracle = oracle
        self.decision_dim = decision_dim
        self.num_samples = num_samples
        self.sigma = sigma
        self.antithetic = antithetic
        self._noise_override: torch.Tensor | None = None

    @contextmanager
    def fixed_noise(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        previous = self._noise_override
        self._noise_override = torch.randn(
            self.num_samples,
            self.decision_dim,
            device=device,
            dtype=dtype,
        )
        try:
            yield
        finally:
            self._noise_override = previous

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        if self._noise_override is None:
            noise = torch.randn(
                self.num_samples,
                self.decision_dim,
                device=scores.device,
                dtype=scores.dtype,
            )
        else:
            noise = self._noise_override.to(device=scores.device, dtype=scores.dtype)
        return _PerturbedOptimizerFunction.apply(scores, noise, self.sigma, self.antithetic, self.oracle)


class TopKOracle:
    """Discrete linear maximization oracle over k-hot vectors."""

    def __init__(self, k: int) -> None:
        self.k = k

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        indices = scores.topk(self.k, dim=-1).indices
        output = torch.zeros_like(scores)
        return output.scatter_(-1, indices, 1.0)


class PerturbedTopKLayer(PerturbedOptimizerLayer):
    """Differentiable perturbed top-k optimizer."""

    def __init__(
        self,
        decision_dim: int,
        k: int,
        num_samples: int = 16,
        sigma: float = 1.0,
        antithetic: bool = True,
    ) -> None:
        super().__init__(
            oracle=TopKOracle(k),
            decision_dim=decision_dim,
            num_samples=num_samples,
            sigma=sigma,
            antithetic=antithetic,
        )
        self.k = k

    def objective(self, decisions: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        return (costs * decisions).sum(dim=-1)

    def regret(self, pred_decisions: torch.Tensor, true_costs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            true_decisions = self.oracle(-true_costs)
        return self.objective(pred_decisions, true_costs) - self.objective(true_decisions, true_costs)


class BipartiteMatchingOracle:
    """Hungarian linear maximization oracle for square bipartite matching."""

    def __init__(self, side: int) -> None:
        self.side = side

    def __call__(self, scores: torch.Tensor) -> torch.Tensor:
        from scipy.optimize import linear_sum_assignment

        device = scores.device
        dtype = scores.dtype
        scores_np = scores.detach().cpu().numpy()
        outputs = []
        for row in scores_np:
            matrix = row.reshape(self.side, self.side)
            rows, cols = linear_sum_assignment(-matrix)
            match = np.zeros_like(matrix, dtype=np.float64)
            match[rows, cols] = 1.0
            outputs.append(match.reshape(-1))
        return torch.as_tensor(np.stack(outputs, axis=0), device=device, dtype=dtype)


class PerturbedBipartiteMatchingLayer(PerturbedOptimizerLayer):
    """Differentiable perturbed discrete bipartite matching layer."""

    def __init__(
        self,
        side: int,
        num_samples: int = 16,
        sigma: float = 1.0,
        antithetic: bool = True,
    ) -> None:
        self.side = side
        super().__init__(
            oracle=BipartiteMatchingOracle(side),
            decision_dim=side * side,
            num_samples=num_samples,
            sigma=sigma,
            antithetic=antithetic,
        )

    def objective(self, decisions: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        return (costs * decisions).sum(dim=-1)

    def regret(self, pred_decisions: torch.Tensor, true_costs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            true_decisions = self.oracle(-true_costs)
        return self.objective(pred_decisions, true_costs) - self.objective(true_decisions, true_costs)
