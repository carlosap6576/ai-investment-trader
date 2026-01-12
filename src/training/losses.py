"""
Trading-specific loss functions for hierarchical sentiment classification.

This module implements specialized loss functions that account for:
- Class imbalance (often more HOLD samples than BUY/SELL)
- Trading-specific concerns (false BUY/SELL signals are costly)
- Focal loss for hard example mining
"""

from typing import Dict, Optional, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal Loss down-weights easy examples and focuses training on hard examples.
    This is particularly useful when one class (e.g., HOLD) dominates the dataset.

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Per-class weights. If None, uses uniform weights.
        gamma: Focusing parameter. Higher gamma = more focus on hard examples.
               gamma=0 is equivalent to standard cross-entropy.
               gamma=2 is commonly used.
        reduction: 'none', 'mean', or 'sum'
    """

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Logits of shape (batch_size, num_classes)
            targets: Class indices of shape (batch_size,)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TradingSignalLoss(nn.Module):
    """
    Specialized loss function for trading signal classification.

    This loss combines:
    1. Focal Loss for handling class imbalance
    2. Directional penalty for wrong BUY/SELL predictions (higher cost)
    3. Confidence calibration to avoid overconfident wrong predictions

    In trading:
    - Predicting HOLD when it should be BUY/SELL = missed opportunity (bad)
    - Predicting BUY/SELL when it should be HOLD = false signal (worse)
    - Predicting BUY when it should be SELL = opposite direction (worst)
    """

    def __init__(
        self,
        class_weights: Optional[Tensor] = None,
        focal_gamma: float = 2.0,
        directional_penalty: float = 1.5,
        confidence_weight: float = 0.1,
    ):
        """
        Initialize trading signal loss.

        Args:
            class_weights: Per-class weights [SELL, HOLD, BUY]. If None, computed from data.
            focal_gamma: Focusing parameter for focal loss
            directional_penalty: Extra penalty for wrong direction (BUY↔SELL)
            confidence_weight: Weight for confidence calibration term
        """
        super().__init__()
        self.focal_gamma = focal_gamma
        self.directional_penalty = directional_penalty
        self.confidence_weight = confidence_weight

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        # Penalty matrix for directional errors
        # penalty[pred][actual] = extra penalty
        # BUY (2) predicted when SELL (0) actual = highest penalty
        # SELL (0) predicted when BUY (2) actual = highest penalty
        penalty_matrix = torch.tensor([
            [0.0, 0.5, 1.5],  # Predicted SELL: 0 if correct, 0.5 if HOLD, 1.5 if BUY
            [0.5, 0.0, 0.5],  # Predicted HOLD: 0.5 if SELL, 0 if correct, 0.5 if BUY
            [1.5, 0.5, 0.0],  # Predicted BUY: 1.5 if SELL, 0.5 if HOLD, 0 if correct
        ])
        self.register_buffer('penalty_matrix', penalty_matrix)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        return_components: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Compute trading signal loss.

        Args:
            logits: Model outputs of shape (batch_size, 3)
            targets: Ground truth labels of shape (batch_size,)
            return_components: If True, return dict with loss components

        Returns:
            Total loss (or dict with components if return_components=True)
        """
        batch_size = logits.shape[0]
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

        # 1. Focal Loss component
        if self.class_weights is not None:
            focal_loss = FocalLoss(
                alpha=self.class_weights,
                gamma=self.focal_gamma,
            )
        else:
            focal_loss = FocalLoss(gamma=self.focal_gamma)

        base_loss = focal_loss(logits, targets)

        # 2. Directional penalty
        # Add extra penalty for wrong direction predictions
        directional_loss = torch.zeros(1, device=logits.device)
        for i in range(batch_size):
            pred_idx = int(predictions[i].item())
            target_idx = int(targets[i].item())
            penalty = self.penalty_matrix[pred_idx, target_idx]
            directional_loss = directional_loss + penalty

        directional_loss = directional_loss / batch_size * self.directional_penalty

        # 3. Confidence calibration
        # Penalize high confidence on wrong predictions
        correct_mask = (predictions == targets).float()
        max_probs = probs.max(dim=-1).values

        # For correct predictions: no penalty
        # For wrong predictions: penalty proportional to confidence
        confidence_penalty = ((1 - correct_mask) * max_probs).mean()
        confidence_loss = self.confidence_weight * confidence_penalty

        total_loss = base_loss + directional_loss + confidence_loss

        if return_components:
            return {
                'total': total_loss,
                'focal': base_loss,
                'directional': directional_loss,
                'confidence': confidence_loss,
            }

        return total_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing for regularization.

    Instead of hard labels [0, 0, 1], uses soft labels like [0.05, 0.05, 0.9].
    This prevents the model from becoming overconfident.
    """

    def __init__(
        self,
        num_classes: int = 3,
        smoothing: float = 0.1,
        reduction: str = 'mean',
    ):
        """
        Initialize label smoothing loss.

        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor (0.1 means 10% of probability mass is distributed)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute label smoothing loss.

        Args:
            inputs: Logits of shape (batch_size, num_classes)
            targets: Class indices of shape (batch_size,)

        Returns:
            Label smoothing loss value
        """
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smoothed labels
        with torch.no_grad():
            smooth_labels = torch.full(
                (targets.size(0), self.num_classes),
                self.smoothing / (self.num_classes - 1),
                device=inputs.device,
            )
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # KL divergence
        loss = -smooth_labels * log_probs

        if self.reduction == 'mean':
            return loss.sum(dim=-1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum(dim=-1)


def compute_class_weights(
    labels: Tensor,
    num_classes: int = 3,
    method: str = 'inverse',
) -> Tensor:
    """
    Compute class weights from label distribution.

    Args:
        labels: Tensor of class indices
        num_classes: Number of classes
        method: Weight computation method
            - 'inverse': 1 / class_count
            - 'effective': (1 - beta^n) / (1 - beta) where n = class count
            - 'sqrt': sqrt(max_count / class_count)

    Returns:
        Tensor of class weights of shape (num_classes,)
    """
    # Count samples per class
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()

    # Avoid division by zero
    counts = counts.clamp(min=1)

    if method == 'inverse':
        # Simple inverse frequency
        weights = 1.0 / counts
        weights = weights / weights.sum() * num_classes  # Normalize

    elif method == 'effective':
        # Effective number of samples (for long-tail distributions)
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes

    elif method == 'sqrt':
        # Square root of inverse frequency (softer weighting)
        max_count = counts.max()
        weights = torch.sqrt(max_count / counts)
        weights = weights / weights.sum() * num_classes

    else:
        raise ValueError(f"Unknown method: {method}")

    return weights


def create_trading_loss(
    class_weights: Optional[Tensor] = None,
    focal_gamma: float = 2.0,
    directional_penalty: float = 1.5,
) -> TradingSignalLoss:
    """
    Factory function to create trading signal loss.

    Args:
        class_weights: Per-class weights
        focal_gamma: Focal loss focusing parameter
        directional_penalty: Penalty for wrong direction predictions

    Returns:
        Configured TradingSignalLoss instance
    """
    return TradingSignalLoss(
        class_weights=class_weights,
        focal_gamma=focal_gamma,
        directional_penalty=directional_penalty,
    )
