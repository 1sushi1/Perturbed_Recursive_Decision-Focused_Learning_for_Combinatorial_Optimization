from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
from torch import nn


class RDFLUnrolled(nn.Module):
    """R-DFL-U: explicit K-step unrolling of x_i = G(F_theta(v, x_{i-1}))."""

    def __init__(self, predictor: nn.Module, optimizer_layer: nn.Module, unroll_steps: int = 10) -> None:
        super().__init__()
        self.predictor = predictor
        self.optimizer_layer = optimizer_layer
        self.unroll_steps = unroll_steps

    def forward(
        self,
        features: torch.Tensor,
        initial_decisions: torch.Tensor | None = None,
        return_costs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self._initial_decisions(features, initial_decisions)
        c = None
        for _ in range(self.unroll_steps):
            c = self.predictor(features, x)
            x = self.optimizer_layer(c)
        if return_costs:
            if c is None:
                c = self.predictor(features, x)
            return x, c
        return x

    def _initial_decisions(
        self,
        features: torch.Tensor,
        initial_decisions: torch.Tensor | None,
    ) -> torch.Tensor:
        if initial_decisions is not None:
            return initial_decisions
        dim = self.optimizer_layer.decision_dim
        return torch.zeros(features.shape[0], dim, device=features.device, dtype=features.dtype)


class RDFLImplicit(nn.Module):
    """R-DFL-I: fixed-point forward pass and implicit equilibrium gradient.

    If phi_theta(x) = G(F_theta(v, x)), the backward hook replaces an incoming
    gradient g by approximately solving a = g + J_phi(x*)^T a. This is the
    vector form of a = (I - J_phi(x*)^T)^-1 g.
    """

    def __init__(
        self,
        predictor: nn.Module,
        optimizer_layer: nn.Module,
        rootfind_steps: int = 30,
        backward_steps: int = 30,
        tolerance: float = 1e-4,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.optimizer_layer = optimizer_layer
        self.rootfind_steps = rootfind_steps
        self.backward_steps = backward_steps
        self.tolerance = tolerance

    @contextmanager
    def _deterministic_predictor(self):
        was_training = self.predictor.training
        self.predictor.eval()
        try:
            yield
        finally:
            self.predictor.train(was_training)

    def forward(
        self,
        features: torch.Tensor,
        initial_decisions: torch.Tensor | None = None,
        return_costs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.optimizer_layer, "fixed_noise"):
            opt_context = self.optimizer_layer.fixed_noise(features.shape[0], features.device, features.dtype)
        else:
            opt_context = nullcontext()

        with opt_context:
            x_star = self._find_equilibrium(features, initial_decisions)

            x_req = x_star.detach().requires_grad_(True)
            with self._deterministic_predictor():
                costs = self.predictor(features, x_req)
                phi = self.optimizer_layer(costs)

        if not torch.is_grad_enabled():
            if return_costs:
                return phi, costs
            return phi

        hook = None

        def implicit_hook(grad: torch.Tensor) -> torch.Tensor:
            nonlocal hook
            if hook is not None:
                hook.remove()
                hook = None
            adjoint = grad
            for _ in range(self.backward_steps):
                j_t_adj = torch.autograd.grad(
                    phi,
                    x_req,
                    adjoint,
                    retain_graph=True,
                    allow_unused=False,
                )[0]
                next_adjoint = grad + j_t_adj
                if (next_adjoint - adjoint).norm() <= self.tolerance * (1.0 + adjoint.norm()):
                    adjoint = next_adjoint
                    break
                adjoint = next_adjoint
            return adjoint

        hook = phi.register_hook(implicit_hook)
        if return_costs:
            return phi, costs
        return phi

    @torch.no_grad()
    def _find_equilibrium(
        self,
        features: torch.Tensor,
        initial_decisions: torch.Tensor | None,
    ) -> torch.Tensor:
        if initial_decisions is None:
            dim = self.optimizer_layer.decision_dim
            x = torch.zeros(features.shape[0], dim, device=features.device, dtype=features.dtype)
        else:
            x = initial_decisions

        for _ in range(self.rootfind_steps):
            with self._deterministic_predictor():
                next_x = self.optimizer_layer(self.predictor(features, x))
            if (next_x - x).norm() <= self.tolerance * (1.0 + x.norm()):
                x = next_x
                break
            x = next_x
        return x
