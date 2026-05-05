from __future__ import annotations

import torch
from torch import nn


class FeedbackMLP(nn.Module):
    """Predicts costs from exogenous features and previous decisions."""

    def __init__(
        self,
        feature_dim: int,
        decision_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.decision_dim = decision_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim + decision_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, decision_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="leaky_relu")
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor, decisions: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or decisions.ndim != 2:
            raise ValueError("features and decisions must be rank-2 tensors")
        if features.shape[0] != decisions.shape[0]:
            raise ValueError("features and decisions must have the same batch size")
        return self.net(torch.cat([features, decisions], dim=-1))


class FeatureMLP(nn.Module):
    """Sequential predictor used by PTO and S-DFL baselines."""

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="leaky_relu")
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must be a rank-2 tensor")
        return self.net(features)
