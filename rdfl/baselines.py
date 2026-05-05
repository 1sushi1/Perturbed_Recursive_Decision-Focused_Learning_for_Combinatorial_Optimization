from __future__ import annotations

import torch
from torch import nn


class PTOPipeline(nn.Module):
    """Predict-then-optimize baseline.

    Training usually uses prediction MSE on costs; evaluation optimizes using
    the predicted costs and measures downstream decision quality.
    """

    def __init__(self, predictor: nn.Module, optimizer_layer: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor
        self.optimizer_layer = optimizer_layer

    def predict_costs(self, features: torch.Tensor) -> torch.Tensor:
        return self.predictor(features)

    def forward(self, features: torch.Tensor, return_costs: bool = False):
        costs = self.predictor(features)
        decisions = self.optimizer_layer(costs)
        if return_costs:
            return decisions, costs
        return decisions


class SDFLModel(nn.Module):
    """Sequential DFL baseline without decision feedback."""

    def __init__(self, predictor: nn.Module, optimizer_layer: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor
        self.optimizer_layer = optimizer_layer

    def forward(self, features: torch.Tensor, return_costs: bool = False):
        costs = self.predictor(features)
        decisions = self.optimizer_layer(costs)
        if return_costs:
            return decisions, costs
        return decisions
