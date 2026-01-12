"""
Trading-specific evaluation metrics.

This module provides metrics tailored for trading signal classification:
- Standard classification metrics (accuracy, F1, precision, recall)
- Trading-specific metrics (directional accuracy, simulated PnL, Sharpe ratio)
- Per-class performance analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0


@dataclass
class TradingMetrics:
    """Complete trading signal evaluation metrics."""
    # Classification metrics
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0

    # Per-class metrics
    per_class: Dict[str, ClassMetrics] = field(default_factory=dict)

    # Trading-specific metrics
    directional_accuracy: float = 0.0  # How often we correctly predict direction
    signal_accuracy: float = 0.0       # Accuracy on non-HOLD predictions only
    hold_accuracy: float = 0.0         # Accuracy on HOLD predictions only

    # Simulated trading metrics
    simulated_pnl: float = 0.0         # Simulated profit/loss
    win_rate: float = 0.0              # % of trades that were profitable
    avg_win: float = 0.0               # Average gain on winning trades
    avg_loss: float = 0.0              # Average loss on losing trades
    profit_factor: float = 0.0         # Total gains / Total losses
    sharpe_ratio: float = 0.0          # Risk-adjusted return

    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None

    # Raw data for further analysis
    total_samples: int = 0
    correct_predictions: int = 0


class TradingSignalMetrics:
    """
    Evaluator for trading signal classification.

    Computes both standard classification metrics and trading-specific metrics.
    """

    LABELS = ["SELL", "HOLD", "BUY"]
    LABEL_TO_IDX = {"SELL": 0, "HOLD": 1, "BUY": 2}

    def __init__(
        self,
        transaction_cost: float = 0.001,  # 0.1% per trade
        risk_free_rate: float = 0.0,      # For Sharpe ratio
    ):
        """
        Initialize the metrics evaluator.

        Args:
            transaction_cost: Cost per trade as fraction (0.001 = 0.1%)
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate

    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        price_changes: Optional[np.ndarray] = None,
    ) -> TradingMetrics:
        """
        Compute comprehensive trading metrics.

        Args:
            predictions: Predicted labels (0=SELL, 1=HOLD, 2=BUY)
            targets: Ground truth labels
            price_changes: Actual price changes (%) for PnL simulation

        Returns:
            TradingMetrics with all computed metrics
        """
        metrics = TradingMetrics()

        # Basic counts
        metrics.total_samples = len(predictions)
        metrics.correct_predictions = int((predictions == targets).sum())

        # Standard classification metrics
        metrics.accuracy = accuracy_score(targets, predictions)
        metrics.f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
        metrics.f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)

        # Per-class metrics
        precision = precision_score(targets, predictions, average=None, zero_division=0)
        recall = recall_score(targets, predictions, average=None, zero_division=0)
        f1 = f1_score(targets, predictions, average=None, zero_division=0)

        for idx, label in enumerate(self.LABELS):
            support = int((targets == idx).sum())
            metrics.per_class[label] = ClassMetrics(
                precision=float(precision[idx]) if idx < len(precision) else 0.0,
                recall=float(recall[idx]) if idx < len(recall) else 0.0,
                f1=float(f1[idx]) if idx < len(f1) else 0.0,
                support=support,
            )

        # Confusion matrix
        metrics.confusion_matrix = confusion_matrix(targets, predictions, labels=[0, 1, 2])

        # Trading-specific metrics
        metrics.directional_accuracy = self._compute_directional_accuracy(predictions, targets)
        metrics.signal_accuracy = self._compute_signal_accuracy(predictions, targets)
        metrics.hold_accuracy = self._compute_hold_accuracy(predictions, targets)

        # Simulated trading metrics (if price changes provided)
        if price_changes is not None:
            trading_metrics = self._simulate_trading(predictions, price_changes)
            metrics.simulated_pnl = trading_metrics['pnl']
            metrics.win_rate = trading_metrics['win_rate']
            metrics.avg_win = trading_metrics['avg_win']
            metrics.avg_loss = trading_metrics['avg_loss']
            metrics.profit_factor = trading_metrics['profit_factor']
            metrics.sharpe_ratio = trading_metrics['sharpe_ratio']

        return metrics

    def _compute_directional_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """
        Compute directional accuracy: correct direction on BUY/SELL predictions.

        Considers:
        - Predicting BUY when target is BUY = correct
        - Predicting SELL when target is SELL = correct
        - Predicting BUY when target is SELL = wrong (opposite direction)
        - Predicting SELL when target is BUY = wrong (opposite direction)
        - HOLD predictions are excluded
        """
        # Only consider non-HOLD predictions
        non_hold_mask = predictions != 1
        if not non_hold_mask.any():
            return 0.0

        pred_non_hold = predictions[non_hold_mask]
        target_non_hold = targets[non_hold_mask]

        # Correct if same direction (both BUY or both SELL)
        # Also count as correct if target was HOLD (we made a call, wasn't opposite)
        correct = (pred_non_hold == target_non_hold) | (target_non_hold == 1)

        return float(correct.sum() / len(pred_non_hold))

    def _compute_signal_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Compute accuracy on BUY/SELL predictions only."""
        non_hold_mask = predictions != 1
        if not non_hold_mask.any():
            return 0.0

        return float((predictions[non_hold_mask] == targets[non_hold_mask]).mean())

    def _compute_hold_accuracy(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Compute accuracy on HOLD predictions only."""
        hold_mask = predictions == 1
        if not hold_mask.any():
            return 0.0

        return float((predictions[hold_mask] == targets[hold_mask]).mean())

    def _simulate_trading(
        self,
        predictions: np.ndarray,
        price_changes: np.ndarray,
    ) -> Dict[str, float]:
        """
        Simulate trading based on predictions.

        Strategy:
        - BUY prediction: Go long, PnL = price_change - transaction_cost
        - SELL prediction: Go short, PnL = -price_change - transaction_cost
        - HOLD prediction: No trade, PnL = 0

        Args:
            predictions: Model predictions
            price_changes: Actual price changes (%)

        Returns:
            Dict with simulated trading metrics
        """
        returns = []

        for pred, pct_change in zip(predictions, price_changes):
            if pred == 2:  # BUY
                # Long position: gain if price goes up
                ret = pct_change - self.transaction_cost * 100
                returns.append(ret)
            elif pred == 0:  # SELL
                # Short position: gain if price goes down
                ret = -pct_change - self.transaction_cost * 100
                returns.append(ret)
            # HOLD: no trade, no return

        if not returns:
            return {
                'pnl': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
            }

        returns = np.array(returns)

        # Total PnL
        pnl = float(returns.sum())

        # Win rate
        wins = returns > 0
        win_rate = float(wins.mean()) if len(returns) > 0 else 0.0

        # Average win/loss
        avg_win = float(returns[wins].mean()) if wins.any() else 0.0
        avg_loss = float(returns[~wins].mean()) if (~wins).any() else 0.0

        # Profit factor
        total_wins = float(returns[wins].sum()) if wins.any() else 0.0
        total_losses = float(abs(returns[~wins].sum())) if (~wins).any() else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Sharpe ratio (annualized, assuming daily returns)
        if len(returns) > 1 and returns.std() > 0:
            excess_return = returns.mean() - self.risk_free_rate / 252
            sharpe_ratio = float(excess_return / returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0.0

        return {
            'pnl': pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
        }

    def get_classification_report(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> str:
        """Get sklearn classification report as string."""
        return classification_report(
            targets,
            predictions,
            labels=[0, 1, 2],
            target_names=self.LABELS,
            zero_division=0,
        )


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    price_changes: Optional[np.ndarray] = None,
) -> TradingMetrics:
    """
    Convenience function to compute trading metrics.

    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        price_changes: Optional price changes for PnL simulation

    Returns:
        TradingMetrics instance
    """
    evaluator = TradingSignalMetrics()
    return evaluator.evaluate(predictions, targets, price_changes)


def print_metrics_summary(metrics: TradingMetrics) -> None:
    """Print a summary of the metrics to console."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nClassification Metrics:")
    print(f"  Accuracy:     {metrics.accuracy:.2%}")
    print(f"  F1 (macro):   {metrics.f1_macro:.4f}")
    print(f"  F1 (weighted): {metrics.f1_weighted:.4f}")

    print(f"\nPer-Class Performance:")
    for label, class_metrics in metrics.per_class.items():
        print(f"  {label}:")
        print(f"    Precision: {class_metrics.precision:.2%}")
        print(f"    Recall:    {class_metrics.recall:.2%}")
        print(f"    F1:        {class_metrics.f1:.4f}")
        print(f"    Support:   {class_metrics.support}")

    print(f"\nTrading-Specific Metrics:")
    print(f"  Directional Accuracy: {metrics.directional_accuracy:.2%}")
    print(f"  Signal Accuracy:      {metrics.signal_accuracy:.2%}")
    print(f"  Hold Accuracy:        {metrics.hold_accuracy:.2%}")

    if metrics.simulated_pnl != 0.0:
        print(f"\nSimulated Trading:")
        print(f"  Total PnL:      {metrics.simulated_pnl:+.2f}%")
        print(f"  Win Rate:       {metrics.win_rate:.2%}")
        print(f"  Avg Win:        {metrics.avg_win:+.2f}%")
        print(f"  Avg Loss:       {metrics.avg_loss:+.2f}%")
        print(f"  Profit Factor:  {metrics.profit_factor:.2f}")
        print(f"  Sharpe Ratio:   {metrics.sharpe_ratio:.2f}")

    print("\n" + "=" * 60)
