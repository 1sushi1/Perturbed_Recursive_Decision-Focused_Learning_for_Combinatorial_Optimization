from __future__ import annotations

import torch
from torch import nn


def _as_batch_vector(value: float | torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        value = value.to(device=device)
        if value.ndim == 0:
            return value.expand(batch_size)
        return value
    return torch.full((batch_size,), float(value), device=device)


def project_box_sum_interval(
    y: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    sum_lower: float | torch.Tensor,
    sum_upper: float | torch.Tensor,
    iters: int = 60,
) -> torch.Tensor:
    """Projection onto {lower <= x <= upper, sum_lower <= sum(x) <= sum_upper}.

    The bisection is differentiable with respect to the final clamp expression,
    while the branch selecting the active sum constraint is treated piecewise.
    """

    clipped = y.clamp(min=lower, max=upper)
    sums = clipped.sum(dim=-1)
    batch = y.shape[0]
    t_low = _as_batch_vector(sum_lower, batch, y.device)
    t_high = _as_batch_vector(sum_upper, batch, y.device)
    target = sums.clamp(min=t_low, max=t_high)

    lo = (y - upper).amin(dim=-1) - 1.0
    hi = (y - lower).amax(dim=-1) + 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        trial = (y - mid.unsqueeze(-1)).clamp(min=lower, max=upper)
        too_large = trial.sum(dim=-1) > target
        lo = torch.where(too_large, mid, lo)
        hi = torch.where(too_large, hi, mid)
    tau = (lo + hi) / 2.0
    return (y - tau.unsqueeze(-1)).clamp(min=lower, max=upper)


class _BoxSumProjection(torch.autograd.Function):
    """Projection with an active-set KKT backward pass.

    Away from active-set changes, the projection has the KKT form

        x_i = clamp(y_i - tau, lower_i, upper_i).

    If the sum constraint is inactive, dx/dy is the diagonal mask of variables
    not clipped by box bounds. If the lower/upper sum constraint is active,
    free variables share the Lagrange multiplier tau, giving

        dx_F / dy_F = I - 11^T / |F|,

    while bound variables have zero sensitivity. This is the compact
    active-set equivalent of the KKT sensitivity matrix used in the paper.
    """

    @staticmethod
    def forward(
        ctx,
        y: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        sum_lower: torch.Tensor,
        sum_upper: torch.Tensor,
        iters: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            clipped = y.clamp(min=lower, max=upper)
            clipped_sums = clipped.sum(dim=-1)
            target = clipped_sums.clamp(min=sum_lower, max=sum_upper)
            active_sum = (clipped_sums - target).abs() > 1e-7

            lo = (y - upper).amin(dim=-1) - 1.0
            hi = (y - lower).amax(dim=-1) + 1.0
            for _ in range(int(iters)):
                mid = (lo + hi) / 2.0
                trial = (y - mid.unsqueeze(-1)).clamp(min=lower, max=upper)
                too_large = trial.sum(dim=-1) > target
                lo = torch.where(too_large, mid, lo)
                hi = torch.where(too_large, hi, mid)
            tau = (lo + hi) / 2.0
            x = (y - tau.unsqueeze(-1)).clamp(min=lower, max=upper)

        ctx.save_for_backward(x, lower, upper, active_sum)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, lower, upper, active_sum = ctx.saved_tensors
        tol = 1e-7
        free = (x > lower + tol) & (x < upper - tol)
        grad_y = torch.zeros_like(grad_output)

        inactive = ~active_sum
        if inactive.any():
            grad_y[inactive] = grad_output[inactive] * free[inactive].to(grad_output.dtype)

        if active_sum.any():
            free_active = free[active_sum]
            grad_active = grad_output[active_sum]
            count = free_active.sum(dim=-1, keepdim=True).clamp(min=1).to(grad_output.dtype)
            centered = grad_active - (grad_active * free_active).sum(dim=-1, keepdim=True) / count
            grad_y[active_sum] = centered * free_active.to(grad_output.dtype)

        return grad_y, None, None, None, None, None


def project_box_sum_interval_kkt(
    y: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    sum_lower: float | torch.Tensor,
    sum_upper: float | torch.Tensor,
    iters: int = 60,
) -> torch.Tensor:
    """Projection onto box and sum interval with KKT sensitivity backward."""

    batch = y.shape[0]
    t_low = _as_batch_vector(sum_lower, batch, y.device).to(dtype=y.dtype)
    t_high = _as_batch_vector(sum_upper, batch, y.device).to(dtype=y.dtype)
    return _BoxSumProjection.apply(y, lower, upper, t_low, t_high, iters)


class NewsvendorLayer(nn.Module):
    """Continuous R-MPNP optimization layer.

    Solves a strongly convex variant of the paper's linear program:

        min_x c^T x + 0.5 * l2_strength * ||x||^2
        s.t.  sum_lower <= 1^T x <= sum_upper, lower <= x <= upper.
    """

    def __init__(
        self,
        decision_dim: int,
        sum_lower: float,
        sum_upper: float,
        lower: float | torch.Tensor = 0.0,
        upper: float | torch.Tensor = 1.0,
        l2_strength: float = 1.0,
        projection_iters: int = 60,
        kkt_backward: bool = True,
    ) -> None:
        super().__init__()
        self.decision_dim = decision_dim
        self.sum_lower = sum_lower
        self.sum_upper = sum_upper
        self.l2_strength = l2_strength
        self.projection_iters = projection_iters
        self.kkt_backward = kkt_backward
        self.register_buffer("lower", torch.as_tensor(lower).float().expand(decision_dim).clone())
        self.register_buffer("upper", torch.as_tensor(upper).float().expand(decision_dim).clone())

    def forward(self, costs: torch.Tensor) -> torch.Tensor:
        y = -costs / self.l2_strength
        project = project_box_sum_interval_kkt if self.kkt_backward else project_box_sum_interval
        return project(
            y,
            self.lower.to(device=costs.device, dtype=costs.dtype),
            self.upper.to(device=costs.device, dtype=costs.dtype),
            self.sum_lower,
            self.sum_upper,
            self.projection_iters,
        )

    def objective(self, decisions: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        linear = (costs * decisions).sum(dim=-1)
        quad = 0.5 * self.l2_strength * decisions.square().sum(dim=-1)
        return linear + quad

    def regret(self, pred_decisions: torch.Tensor, true_costs: torch.Tensor) -> torch.Tensor:
        true_decisions = self(true_costs)
        return self.objective(pred_decisions, true_costs) - self.objective(true_decisions, true_costs)


class RelaxedTopKLayer(nn.Module):
    """Continuous top-k relaxation: 0 <= x <= 1 and sum(x) = k."""

    def __init__(
        self,
        decision_dim: int,
        k: int,
        l2_strength: float = 1.0,
        projection_iters: int = 60,
        kkt_backward: bool = True,
    ) -> None:
        super().__init__()
        self.decision_dim = decision_dim
        self.k = k
        self.l2_strength = l2_strength
        self.projection_iters = projection_iters
        self.kkt_backward = kkt_backward
        self.register_buffer("lower", torch.zeros(decision_dim))
        self.register_buffer("upper", torch.ones(decision_dim))

    def forward(self, costs: torch.Tensor) -> torch.Tensor:
        y = -costs / self.l2_strength
        project = project_box_sum_interval_kkt if self.kkt_backward else project_box_sum_interval
        return project(
            y,
            self.lower.to(device=costs.device, dtype=costs.dtype),
            self.upper.to(device=costs.device, dtype=costs.dtype),
            float(self.k),
            float(self.k),
            self.projection_iters,
        )

    def objective(self, decisions: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        linear = (costs * decisions).sum(dim=-1)
        quad = 0.5 * self.l2_strength * decisions.square().sum(dim=-1)
        return linear + quad

    def regret(self, pred_decisions: torch.Tensor, true_costs: torch.Tensor) -> torch.Tensor:
        true_decisions = self(true_costs)
        return self.objective(pred_decisions, true_costs) - self.objective(true_decisions, true_costs)


class BipartiteMatchingLayer(nn.Module):
    """Continuous R-BMP optimization layer with projected gradient iterations."""

    def __init__(
        self,
        num_left: int,
        num_right: int | None = None,
        min_matches: float = 1.0,
        l2_strength: float = 0.1,
        step_size: float = 0.2,
        projection_iters: int = 25,
        solver_iters: int = 80,
    ) -> None:
        super().__init__()
        self.num_left = num_left
        self.num_right = num_right or num_left
        self.decision_dim = self.num_left * self.num_right
        self.min_matches = min_matches
        self.l2_strength = l2_strength
        self.step_size = step_size
        self.projection_iters = projection_iters
        self.solver_iters = solver_iters

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        x = z.reshape(batch, self.num_left, self.num_right).clamp(0.0, 1.0)
        for _ in range(self.projection_iters):
            row_scale = torch.clamp(x.sum(dim=2, keepdim=True), min=1.0)
            x = x / row_scale
            col_scale = torch.clamp(x.sum(dim=1, keepdim=True), min=1.0)
            x = x / col_scale
            total = x.sum(dim=(1, 2), keepdim=True)
            deficit = (self.min_matches - total).clamp(min=0.0)
            room = (1.0 - x).clamp(min=0.0)
            room_sum = room.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
            x = (x + deficit * room / room_sum).clamp(0.0, 1.0)
        return x.reshape(batch, self.decision_dim)

    def forward(self, costs: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(-costs)
        z = self._project(z)
        for _ in range(self.solver_iters):
            grad = costs + 2.0 * self.l2_strength * z
            z = self._project(z - self.step_size * grad)
        return z

    def objective(self, decisions: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        linear = (costs * decisions).sum(dim=-1)
        quad = self.l2_strength * decisions.square().sum(dim=-1)
        return linear + quad

    def regret(self, pred_decisions: torch.Tensor, true_costs: torch.Tensor) -> torch.Tensor:
        true_decisions = self(true_costs)
        return self.objective(pred_decisions, true_costs) - self.objective(true_decisions, true_costs)
