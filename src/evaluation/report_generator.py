"""
Report Generator for trading signal classification.

Generates formatted reports from evaluation metrics including:
- Text reports for console output
- Confusion matrix visualization
- Feature importance analysis
"""

from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from .metrics import TradingMetrics


class ReportGenerator:
    """
    Generates formatted evaluation reports.

    Supports multiple output formats:
    - Console text output
    - Structured dict for JSON export
    - Attention weight analysis
    """

    def __init__(self, symbol: str = "UNKNOWN"):
        """
        Initialize report generator.

        Args:
            symbol: Trading symbol for the report
        """
        self.symbol = symbol

    def generate_console_report(
        self,
        metrics: TradingMetrics,
        model_name: str = "HierarchicalSentimentTransformer",
        attention_weights: Optional[Dict[str, float]] = None,
        top_features: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a formatted console report.

        Args:
            metrics: TradingMetrics instance
            model_name: Name of the model
            attention_weights: Optional level importance weights
            top_features: Optional list of most important features

        Returns:
            Formatted string report
        """
        lines = []
        width = 70

        # Header
        lines.append("=" * width)
        lines.append("TRADING SIGNAL CLASSIFIER - EVALUATION REPORT".center(width))
        lines.append("=" * width)
        lines.append("")

        # Metadata
        lines.append("REPORT METADATA")
        lines.append("-" * width)
        lines.append(f"  Symbol:           {self.symbol}")
        lines.append(f"  Model:            {model_name}")
        lines.append(f"  Generated:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Total Samples:    {metrics.total_samples}")
        lines.append("")

        # Classification Metrics
        lines.append("CLASSIFICATION METRICS")
        lines.append("-" * width)
        lines.append(f"  Accuracy:           {metrics.correct_predictions}/{metrics.total_samples} = {metrics.accuracy:.2%}")
        lines.append(f"  F1 Score (macro):   {metrics.f1_macro:.4f}")
        lines.append(f"  F1 Score (weighted): {metrics.f1_weighted:.4f}")
        lines.append("")

        # Per-class performance
        lines.append("PER-CLASS PERFORMANCE")
        lines.append("-" * width)
        lines.append(f"  {'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        lines.append(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
        for label in ["SELL", "HOLD", "BUY"]:
            if label in metrics.per_class:
                cm = metrics.per_class[label]
                lines.append(
                    f"  {label:<10} {cm.precision:<12.2%} {cm.recall:<12.2%} "
                    f"{cm.f1:<12.4f} {cm.support:<10}"
                )
        lines.append("")

        # Confusion Matrix
        if metrics.confusion_matrix is not None:
            lines.append("CONFUSION MATRIX")
            lines.append("-" * width)
            lines.append("  Predicted →")
            lines.append(f"  {'Actual ↓':<12} {'SELL':>10} {'HOLD':>10} {'BUY':>10}")
            lines.append(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
            for i, label in enumerate(["SELL", "HOLD", "BUY"]):
                row = metrics.confusion_matrix[i]
                lines.append(f"  {label:<12} {row[0]:>10} {row[1]:>10} {row[2]:>10}")
            lines.append("")

        # Trading-specific metrics
        lines.append("TRADING-SPECIFIC METRICS")
        lines.append("-" * width)
        lines.append(f"  Directional Accuracy: {metrics.directional_accuracy:.2%}")
        lines.append(f"  Signal Accuracy:      {metrics.signal_accuracy:.2%} (BUY/SELL only)")
        lines.append(f"  Hold Accuracy:        {metrics.hold_accuracy:.2%}")
        lines.append("")

        # Simulated trading results
        if metrics.simulated_pnl != 0.0 or metrics.win_rate != 0.0:
            lines.append("SIMULATED TRADING RESULTS")
            lines.append("-" * width)
            pnl_sign = "+" if metrics.simulated_pnl >= 0 else ""
            lines.append(f"  Total PnL:       {pnl_sign}{metrics.simulated_pnl:.2f}%")
            lines.append(f"  Win Rate:        {metrics.win_rate:.2%}")
            lines.append(f"  Avg Win:         +{metrics.avg_win:.2f}%")
            lines.append(f"  Avg Loss:        {metrics.avg_loss:.2f}%")
            if metrics.profit_factor != float('inf'):
                lines.append(f"  Profit Factor:   {metrics.profit_factor:.2f}")
            else:
                lines.append(f"  Profit Factor:   N/A (no losses)")
            lines.append(f"  Sharpe Ratio:    {metrics.sharpe_ratio:.2f}")
            lines.append("")

        # Attention weights (level importance)
        if attention_weights:
            lines.append("LEVEL IMPORTANCE (from attention)")
            lines.append("-" * width)
            total = sum(attention_weights.values())
            for level, weight in sorted(attention_weights.items(), key=lambda x: -x[1]):
                pct = (weight / total * 100) if total > 0 else 0
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                lines.append(f"  {level:<10} {bar} {pct:.1f}%")
            lines.append("")

        # Top features
        if top_features:
            lines.append("TOP FEATURES")
            lines.append("-" * width)
            for i, feature in enumerate(top_features[:10], 1):
                lines.append(f"  {i:>2}. {feature}")
            lines.append("")

        # Footer
        lines.append("=" * width)

        return "\n".join(lines)

    def generate_dict_report(
        self,
        metrics: TradingMetrics,
        model_name: str = "HierarchicalSentimentTransformer",
        attention_weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Generate a dictionary report for JSON export.

        Args:
            metrics: TradingMetrics instance
            model_name: Name of the model
            attention_weights: Optional level importance weights

        Returns:
            Dictionary with all metrics
        """
        report = {
            "metadata": {
                "symbol": self.symbol,
                "model": model_name,
                "generated_at": datetime.now().isoformat(),
                "total_samples": metrics.total_samples,
            },
            "classification": {
                "accuracy": metrics.accuracy,
                "f1_macro": metrics.f1_macro,
                "f1_weighted": metrics.f1_weighted,
                "correct_predictions": metrics.correct_predictions,
            },
            "per_class": {
                label: asdict(class_metrics)
                for label, class_metrics in metrics.per_class.items()
            },
            "trading": {
                "directional_accuracy": metrics.directional_accuracy,
                "signal_accuracy": metrics.signal_accuracy,
                "hold_accuracy": metrics.hold_accuracy,
            },
            "simulated_trading": {
                "pnl": metrics.simulated_pnl,
                "win_rate": metrics.win_rate,
                "avg_win": metrics.avg_win,
                "avg_loss": metrics.avg_loss,
                "profit_factor": metrics.profit_factor,
                "sharpe_ratio": metrics.sharpe_ratio,
            },
        }

        if metrics.confusion_matrix is not None:
            report["confusion_matrix"] = metrics.confusion_matrix.tolist()

        if attention_weights:
            report["level_importance"] = attention_weights

        return report

    def generate_sample_predictions_report(
        self,
        samples: List[Dict],
        num_samples: int = 5,
    ) -> str:
        """
        Generate a report showing sample predictions.

        Args:
            samples: List of dicts with 'text', 'prediction', 'confidence', 'actual'
            num_samples: Number of samples to show

        Returns:
            Formatted string with sample predictions
        """
        lines = []
        lines.append("")
        lines.append("SAMPLE PREDICTIONS")
        lines.append("-" * 70)
        lines.append(f"Showing {min(num_samples, len(samples))} sample predictions:")
        lines.append("")

        for i, sample in enumerate(samples[:num_samples], 1):
            text = sample.get('text', 'N/A')
            if len(text) > 60:
                text = text[:57] + "..."

            pred = sample.get('prediction', 'N/A')
            conf = sample.get('confidence', 0.0)
            actual = sample.get('actual', 'N/A')

            # Determine if correct
            correct = "✓" if pred == actual else "✗"

            lines.append(f"  Sample {i}:")
            lines.append(f"    Text:       {text}")
            lines.append(f"    Prediction: {pred} ({conf:.1%} confident)")
            lines.append(f"    Actual:     {actual} {correct}")
            lines.append("")

        return "\n".join(lines)


def create_report_generator(symbol: str) -> ReportGenerator:
    """
    Factory function to create a report generator.

    Args:
        symbol: Trading symbol

    Returns:
        Configured ReportGenerator instance
    """
    return ReportGenerator(symbol=symbol)


def format_confusion_matrix(cm: np.ndarray, labels: List[str] = None) -> str:
    """
    Format a confusion matrix as a string.

    Args:
        cm: Confusion matrix array
        labels: Class labels

    Returns:
        Formatted string
    """
    if labels is None:
        labels = ["SELL", "HOLD", "BUY"]

    lines = []
    lines.append("Predicted →")
    header = f"{'Actual ↓':<12}" + "".join(f"{l:>10}" for l in labels)
    lines.append(header)
    lines.append("-" * len(header))

    for i, label in enumerate(labels):
        row = f"{label:<12}" + "".join(f"{cm[i, j]:>10}" for j in range(len(labels)))
        lines.append(row)

    return "\n".join(lines)
