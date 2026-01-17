"""
Model Evaluator for the FinBERT Sentiment Strategy.

Provides comprehensive evaluation of trained hierarchical sentiment models,
including accuracy metrics, trading simulation, and detailed reports.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from strategies.finbert_sentiment.constants import DATASETS_DIR, LABEL_NAMES
from strategies.finbert_sentiment.evaluation.config import TestConfig


class ModelEvaluator:
    """
    Evaluates trained hierarchical sentiment trading signal classifiers.

    Loads saved model weights and evaluates on test data, generating
    comprehensive reports with accuracy metrics, trading performance,
    and optional AI-powered summaries.

    Attributes:
        config: TestConfig containing evaluation parameters
        model: Loaded model instance (set after load_model)
        test_data: Tuple of test sequences, labels, timestamps, news_items
        results: Dictionary of evaluation results (set after run)
    """

    def __init__(self, config: TestConfig) -> None:
        """
        Initialize the ModelEvaluator.

        Args:
            config: TestConfig instance with evaluation parameters.
                   Must include symbol, thresholds, architecture params.
        """
        self.config = config
        self.model = None
        self.test_data: Optional[Tuple] = None
        self.results: Optional[Dict[str, Any]] = None

    def run(self) -> Dict[str, Any]:
        """
        Execute the full evaluation pipeline.

        Orchestrates model loading, data loading, evaluation metrics
        computation, and report generation.

        Returns:
            Dictionary containing all evaluation results:
                - accuracy: Overall accuracy (float)
                - f1_macro: Macro F1 score (float)
                - f1_weighted: Weighted F1 score (float)
                - directional_accuracy: Accuracy on BUY/SELL signals (float)
                - per_class_metrics: Dict with precision/recall/f1 per class
                - confusion_matrix: 3x3 numpy array
                - trading_metrics: Dict with PnL, Sharpe, drawdown, win_rate
                - attention_analysis: Dict with level importance weights
                - feature_importance: Dict mapping feature names to scores
                - calibration_metrics: Dict with Brier score, ECE, confidence
                - sample_predictions: List of sample prediction dicts
                - report_lines: List of report text lines
        """
        self._check_required_files()
        self._load_model()
        self._load_test_data()
        self.results = self._evaluate()
        return self.results

    def _check_required_files(self) -> None:
        """
        Check if all required files exist.

        Raises:
            FileNotFoundError: If model or data file is missing.
            ValueError: If data file is empty.
        """
        import json
        import os

        errors = []

        if not os.path.exists(self.config.model_file):
            errors.append(f"Model file not found: {self.config.model_file}")
            errors.append(
                f"  -> Run 'python -m cli.finbert_sentiment.train -s {self.config.symbol}' first"
            )

        if not os.path.exists(self.config.data_file):
            errors.append(f"Data file not found: {self.config.data_file}")
            errors.append(
                f"  -> Run 'python -m cli.finbert_sentiment.download -s {self.config.symbol}' first"
            )
        else:
            with open(self.config.data_file, "r") as f:
                data = json.load(f)
            if len(data) == 0:
                errors.append(f"Data file is empty: {self.config.data_file}")
                errors.append(
                    f"  -> Run 'python -m cli.finbert_sentiment.download -s {self.config.symbol}' to populate"
                )

        if errors:
            error_msg = "\n".join(errors)
            raise FileNotFoundError(f"Missing required files:\n{error_msg}")

    def load_model_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Load model architecture metadata from companion JSON file.

        The metadata file is created by training and contains:
        - hidden_dim: Model internal dimension
        - num_layers: Number of transformer layers
        - seq_length: Temporal sequence length
        - symbol: Trading symbol
        - buy_threshold: Buy threshold used during training
        - sell_threshold: Sell threshold used during training

        Returns:
            Metadata dictionary if found, None otherwise.
        """
        import json
        import os

        metadata_file = self.config.model_file.replace(".pth", "_metadata.json")

        if not os.path.exists(metadata_file):
            return None

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            return metadata
        except Exception:
            return None

    def _load_model(self) -> None:
        """
        Load trained hierarchical model from disk.

        Uses config parameters for architecture (hidden_dim, num_layers, seq_length).
        Sets self.model to the loaded model instance.

        Raises:
            ImportError: If model dependencies are not available.
        """
        import torch
        from src.models.transformer import (
            HierarchicalSentimentTransformer,
        )

        self.model = HierarchicalSentimentTransformer(
            hidden_dim=self.config.hidden_dim,
            num_temporal_layers=self.config.num_layers,
            sequence_length=self.config.seq_length,
        )
        self.model.load_state_dict(
            torch.load(self.config.model_file, weights_only=True)
        )

    def _load_test_data(self) -> None:
        """
        Load test portion of the dataset for hierarchical model.

        Converts raw JSON data to NewsItem objects, builds temporal sequences,
        and splits to get only the test portion based on config.split.

        Sets self.test_data to (test_sequences, test_labels, test_timestamps, news_items).

        Raises:
            ValueError: If no valid news items found in data.
        """
        import json

        import numpy as np
        from src.data.schemas import NewsItem
        from src.features.feature_builder import (
            TemporalFeatureBuilder,
            create_label_function,
        )

        with open(self.config.data_file, "r") as f:
            data = json.load(f)

        # Convert to NewsItem objects
        news_items = []
        for item in data:
            try:
                news_item = NewsItem(
                    headline=item.get("title", ""),
                    summary=item.get("summary", ""),
                    timestamp=datetime.fromisoformat(
                        item["pubDate"].replace("Z", "+00:00")
                    ),
                    source=item.get("source", "unknown"),
                    url=item.get("url", ""),
                    level=item.get("level", "TICKER"),
                    sentiment_score=item.get("sentiment_score", 0.0),
                    sentiment_label=item.get("sentiment_label", "neutral"),
                    price=item.get("price", 0.0),
                    future_price=item.get("future_price"),
                    price_change_pct=item.get("percentage", 0.0),
                )
                news_items.append(news_item)
            except (KeyError, ValueError):
                continue

        if len(news_items) == 0:
            raise ValueError(
                f"No valid news items found in {self.config.data_file}. "
                f"Re-run: python -m cli.finbert_sentiment.download -s {self.config.symbol}"
            )

        # Create feature builder
        feature_builder = TemporalFeatureBuilder(
            target_ticker=self.config.symbol,
            sequence_length=self.config.seq_length,
        )

        # Create label function
        label_func = create_label_function(
            buy_threshold=self.config.buy_threshold,
            sell_threshold=self.config.sell_threshold,
        )

        # Build sequences
        sequences, labels, timestamps = feature_builder.build_training_data(
            news_items=news_items,
            label_func=label_func,
        )

        # Get only test portion
        split_index = int(self.config.split * len(sequences))
        test_sequences = sequences[split_index:]
        test_labels = labels[split_index:]
        test_timestamps = timestamps[split_index:]

        self.test_data = (test_sequences, test_labels, test_timestamps, news_items)

    def _evaluate(self) -> Dict[str, Any]:
        """
        Comprehensive evaluation of hierarchical model with detailed report.

        Computes all metrics and generates the full evaluation report.

        Returns:
            Dictionary containing all evaluation results and report lines.
        """
        import numpy as np
        import torch
        from sklearn.metrics import (
            brier_score_loss,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        test_sequences, test_labels, test_timestamps, news_items = self.test_data

        # Collect all predictions and probabilities
        all_predictions = []
        all_actuals = []
        all_probs = []
        all_confidences = []
        attention_weights_sum = {"ticker": 0.0, "sector": 0.0, "market": 0.0}
        sample_predictions = []

        self.model.eval()
        with torch.no_grad():
            for i in range(len(test_sequences)):
                seq = (
                    torch.from_numpy(test_sequences[i : i + 1]).float().to(self.model.device)
                )
                target = test_labels[i]

                logits = self.model(seq)
                probs = logits.softmax(dim=-1).cpu().numpy()[0]
                predicted = int(np.argmax(probs))
                confidence = float(probs[predicted])

                all_predictions.append(predicted)
                all_actuals.append(target)
                all_probs.append(probs)
                all_confidences.append(confidence)

                # Collect attention weights
                importance = self.model.get_level_importance()
                if importance:
                    attention_weights_sum["ticker"] += importance.get("ticker", 0.33)
                    attention_weights_sum["sector"] += importance.get("sector", 0.33)
                    attention_weights_sum["market"] += importance.get("market", 0.33)

                # Collect sample predictions
                if len(sample_predictions) < self.config.samples:
                    timestamp = test_timestamps[i] if i < len(test_timestamps) else None
                    news_item = news_items[i] if i < len(news_items) else None
                    headline = ""
                    if news_item:
                        headline = (
                            news_item.headline[:50] + "..."
                            if len(news_item.headline) > 50
                            else news_item.headline
                        )
                    sample_predictions.append(
                        {
                            "ticker": self.config.symbol,
                            "date": (
                                timestamp.strftime("%Y-%m-%d %H:%M")
                                if timestamp
                                else "N/A"
                            ),
                            "prediction": LABEL_NAMES[predicted],
                            "actual": LABEL_NAMES[target],
                            "confidence": confidence,
                            "probs": probs.tolist(),
                            "correct": predicted == target,
                            "headline": headline if headline else "N/A",
                        }
                    )

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        all_probs = np.array(all_probs)
        all_confidences = np.array(all_confidences)

        # Calculate overall metrics
        n_samples = len(all_predictions)
        accuracy = float(np.mean(all_predictions == all_actuals))
        f1_macro = f1_score(all_actuals, all_predictions, average="macro", zero_division=0)
        f1_weighted = f1_score(
            all_actuals, all_predictions, average="weighted", zero_division=0
        )

        # Per-class metrics
        precision_per_class = precision_score(
            all_actuals, all_predictions, average=None, labels=[0, 1, 2], zero_division=0
        )
        recall_per_class = recall_score(
            all_actuals, all_predictions, average=None, labels=[0, 1, 2], zero_division=0
        )
        f1_per_class = f1_score(
            all_actuals, all_predictions, average=None, labels=[0, 1, 2], zero_division=0
        )
        support_per_class = [int(np.sum(all_actuals == i)) for i in range(3)]

        # Directional accuracy
        non_hold_mask = all_predictions != 1
        if np.any(non_hold_mask):
            directional_correct = (
                all_predictions[non_hold_mask] == all_actuals[non_hold_mask]
            ) | (all_actuals[non_hold_mask] == 1)
            directional_accuracy = float(np.mean(directional_correct))
        else:
            directional_accuracy = 0.0

        # Trading simulation
        trading_metrics = self._compute_trading_metrics(
            all_predictions, news_items, n_samples
        )

        # Attention analysis
        attention_analysis = self._compute_attention_analysis(
            attention_weights_sum, n_samples
        )

        # Feature importance
        feature_importance = self._compute_feature_importance(test_sequences)

        # Calibration metrics
        correct_mask = all_predictions == all_actuals
        calibration_metrics = self._compute_calibration_metrics(
            all_probs, all_actuals, all_confidences, correct_mask
        )

        # Confusion matrix
        cm = confusion_matrix(all_actuals, all_predictions, labels=[0, 1, 2])

        # Build per-class metrics dict
        per_class_metrics = {
            "precision": precision_per_class.tolist(),
            "recall": recall_per_class.tolist(),
            "f1": f1_per_class.tolist(),
            "support": support_per_class,
        }

        # Generate report
        report_lines = self._generate_report(
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            directional_accuracy=directional_accuracy,
            per_class_metrics=per_class_metrics,
            confusion_matrix=cm,
            trading_metrics=trading_metrics,
            attention_analysis=attention_analysis,
            feature_importance=feature_importance,
            sample_predictions=sample_predictions,
            calibration_metrics=calibration_metrics,
            n_samples=n_samples,
        )

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "directional_accuracy": directional_accuracy,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm,
            "trading_metrics": trading_metrics,
            "attention_analysis": attention_analysis,
            "feature_importance": feature_importance,
            "calibration_metrics": calibration_metrics,
            "sample_predictions": sample_predictions,
            "report_lines": report_lines,
            "n_samples": n_samples,
        }

    def _compute_trading_metrics(
        self,
        predictions: "np.ndarray",
        news_items: List,
        n_samples: int,
    ) -> Dict[str, float]:
        """
        Compute simulated trading performance metrics.

        Args:
            predictions: Array of predicted labels (0=SELL, 1=HOLD, 2=BUY)
            news_items: List of NewsItem objects with price change info
            n_samples: Total number of samples

        Returns:
            Dictionary with PnL, Sharpe ratio, max drawdown, win rate, total trades.
        """
        import numpy as np

        simulated_pnl = 0.0
        win_count = 0
        loss_count = 0
        returns = []

        for i in range(len(predictions)):
            pred = predictions[i]
            if pred != 1:  # Only trade on BUY/SELL signals
                if i < len(news_items) and news_items[i].price_change_pct is not None:
                    price_change = news_items[i].price_change_pct
                else:
                    price_change = 0.0

                if pred == 2:  # BUY
                    ret = price_change - 0.1  # 0.1% transaction cost
                else:  # SELL
                    ret = -price_change - 0.1

                returns.append(ret)
                simulated_pnl += ret
                if ret > 0:
                    win_count += 1
                else:
                    loss_count += 1

        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0

        # Sharpe ratio
        if len(returns) > 1:
            returns_arr = np.array(returns)
            std = np.std(returns_arr)
            sharpe_ratio = (
                float(np.mean(returns_arr) / std * np.sqrt(252)) if std > 0 else 0.0
            )
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        if len(returns) > 0:
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        else:
            max_drawdown = 0.0

        return {
            "simulated_pnl": simulated_pnl,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades,
        }

    def _compute_attention_analysis(
        self,
        attention_weights_sum: Dict[str, float],
        n_samples: int,
    ) -> Dict[str, float]:
        """
        Compute normalized attention weights for each hierarchical level.

        Args:
            attention_weights_sum: Sum of attention weights per level
            n_samples: Number of samples for normalization

        Returns:
            Dictionary with normalized weights per level.
        """
        total_attention = sum(attention_weights_sum.values())
        if total_attention > 0 and n_samples > 0:
            return {
                "ticker_weight": attention_weights_sum["ticker"] / n_samples,
                "sector_weight": attention_weights_sum["sector"] / n_samples,
                "market_weight": attention_weights_sum["market"] / n_samples,
                "cross_weight": 0.0,
                "price_weight": 0.0,
            }
        else:
            return {
                "ticker_weight": 0.33,
                "sector_weight": 0.33,
                "market_weight": 0.33,
                "cross_weight": 0.0,
                "price_weight": 0.0,
            }

    def _compute_feature_importance(
        self, test_sequences: "np.ndarray"
    ) -> Dict[str, float]:
        """
        Compute feature importance based on variance-based attribution.

        Features with higher variance across sequences are more discriminative.

        Args:
            test_sequences: Array of shape (n_samples, seq_length, n_features)

        Returns:
            Dictionary mapping feature names to importance scores (0-1).
        """
        import numpy as np

        feature_names = [
            # Ticker features (0-11)
            "ticker_mean_sentiment",
            "ticker_sentiment_std",
            "ticker_sentiment_skew",
            "ticker_news_volume",
            "ticker_momentum_1d",
            "ticker_momentum_5d",
            "ticker_max_sentiment",
            "ticker_min_sentiment",
            "ticker_positive_ratio",
            "ticker_negative_ratio",
            "ticker_news_recency",
            "ticker_source_diversity",
            # Sector features (12-21)
            "sector_mean_sentiment",
            "sector_sentiment_breadth",
            "sector_news_volume",
            "sector_sentiment_dispersion",
            "sector_leader_sentiment",
            "sector_momentum",
            "sector_relative_strength",
            "sector_news_concentration",
            "sector_peer_sentiment",
            "sector_laggard_sentiment",
            # Market features (22-31)
            "market_sentiment_index",
            "market_news_volume",
            "market_fed_sentiment",
            "market_economic_sentiment",
            "market_geopolitical_sentiment",
            "market_fear_index",
            "market_momentum",
            "market_breadth_sentiment",
            "market_vix_level",
            "market_regime",
            # Cross-level features (32-39)
            "cross_ticker_vs_sector",
            "cross_ticker_vs_market",
            "cross_sector_vs_market",
            "cross_ticker_sector_corr",
            "cross_ticker_market_beta",
            "cross_divergence_score",
            "cross_relative_attention",
            "cross_sentiment_surprise",
        ]

        if len(test_sequences) > 0:
            avg_features = np.mean(test_sequences, axis=1)
            feature_variance = np.var(avg_features, axis=0)

            if np.max(feature_variance) > 0:
                feature_variance = feature_variance / np.max(feature_variance)

            importance = {}
            for i, name in enumerate(feature_names):
                if i < len(feature_variance):
                    importance[name] = float(feature_variance[i])
                else:
                    importance[name] = 0.0
        else:
            importance = {name: 0.0 for name in feature_names}

        return importance

    def _compute_calibration_metrics(
        self,
        all_probs: "np.ndarray",
        all_actuals: "np.ndarray",
        all_confidences: "np.ndarray",
        correct_mask: "np.ndarray",
    ) -> Dict[str, float]:
        """
        Compute calibration metrics for model predictions.

        Args:
            all_probs: Probability distributions for each prediction
            all_actuals: Ground truth labels
            all_confidences: Max probability (confidence) for each prediction
            correct_mask: Boolean mask of correct predictions

        Returns:
            Dictionary with Brier score, ECE, confidence when correct/wrong.
        """
        import numpy as np
        from sklearn.metrics import brier_score_loss

        # Confidence analysis
        conf_when_correct = (
            float(np.mean(all_confidences[correct_mask]))
            if np.any(correct_mask)
            else 0.0
        )
        conf_when_wrong = (
            float(np.mean(all_confidences[~correct_mask]))
            if np.any(~correct_mask)
            else 0.0
        )

        # Brier score (multiclass, one-vs-rest average)
        brier_scores = []
        for c in range(3):
            y_true_binary = (all_actuals == c).astype(float)
            y_prob = all_probs[:, c]
            brier_scores.append(brier_score_loss(y_true_binary, y_prob))
        brier_score = float(np.mean(brier_scores))

        # Expected Calibration Error
        ece = self._compute_expected_calibration_error(all_confidences, correct_mask)

        return {
            "brier_score": brier_score,
            "ece": ece,
            "conf_when_correct": conf_when_correct,
            "conf_when_wrong": conf_when_wrong,
        }

    def _compute_expected_calibration_error(
        self,
        confidences: "np.ndarray",
        correct_mask: "np.ndarray",
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Args:
            confidences: Array of prediction confidences
            correct_mask: Boolean array of correct predictions
            n_bins: Number of bins for calibration

        Returns:
            ECE value (lower is better, 0 is perfect calibration).
        """
        import numpy as np

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (
                confidences <= bin_boundaries[i + 1]
            )
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(correct_mask[in_bin])
                ece += prop_in_bin * np.abs(avg_accuracy - avg_confidence)

        return float(ece)

    def _generate_report(
        self,
        accuracy: float,
        f1_macro: float,
        f1_weighted: float,
        directional_accuracy: float,
        per_class_metrics: Dict[str, List],
        confusion_matrix: "np.ndarray",
        trading_metrics: Dict[str, float],
        attention_analysis: Dict[str, float],
        feature_importance: Dict[str, float],
        sample_predictions: List[Dict],
        calibration_metrics: Dict[str, float],
        n_samples: int,
    ) -> List[str]:
        """
        Generate the full evaluation report as a list of lines.

        Args:
            accuracy: Overall accuracy
            f1_macro: Macro-averaged F1 score
            f1_weighted: Weighted F1 score
            directional_accuracy: Accuracy on directional (non-HOLD) predictions
            per_class_metrics: Dict with precision/recall/f1/support per class
            confusion_matrix: 3x3 confusion matrix
            trading_metrics: Dict with trading simulation results
            attention_analysis: Dict with hierarchical level weights
            feature_importance: Dict with feature importance scores
            sample_predictions: List of sample prediction dicts
            calibration_metrics: Dict with calibration scores
            n_samples: Total number of test samples

        Returns:
            List of report lines (strings).
        """
        report = []
        report.append("")
        report.append("=" * 80)
        report.append(
            "HIERARCHICAL SENTIMENT TRADING SIGNAL CLASSIFIER - EVALUATION REPORT"
        )
        report.append("=" * 80)
        report.append("")

        # Section 1: Overall Performance
        report.append("## 1. OVERALL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"  Accuracy:              {accuracy:.4f} ({accuracy*100:.2f}%)")
        report.append(f"  F1 Score (Macro):      {f1_macro:.4f}")
        report.append(f"  F1 Score (Weighted):   {f1_weighted:.4f}")
        report.append(f"  Directional Accuracy:  {directional_accuracy:.4f}")
        report.append("")

        # Section 2: Per-Class Performance
        report.append("## 2. PER-CLASS PERFORMANCE")
        report.append("-" * 40)
        report.append("  Class     | Precision | Recall  | F1-Score | Support")
        report.append("  ----------|-----------|---------|----------|--------")
        for i, label in enumerate(LABEL_NAMES):
            prec = per_class_metrics["precision"][i]
            rec = per_class_metrics["recall"][i]
            f1 = per_class_metrics["f1"][i]
            sup = per_class_metrics["support"][i]
            report.append(f"  {label:<10}| {prec:.4f}    | {rec:.4f}  | {f1:.4f}   | {sup}")
        report.append("")

        # Section 3: Confusion Matrix
        report.append("## 3. CONFUSION MATRIX")
        report.append("-" * 40)
        report.append("  Predicted ->")
        report.append(f"  {'Actual v':<12} {'SELL':>10} {'HOLD':>10} {'BUY':>10}")
        report.append(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        for i, label in enumerate(LABEL_NAMES):
            report.append(
                f"  {label:<12} {confusion_matrix[i, 0]:>10} "
                f"{confusion_matrix[i, 1]:>10} {confusion_matrix[i, 2]:>10}"
            )
        report.append("")

        # Section 4: Trading Performance
        report.append("## 4. TRADING PERFORMANCE (Simulated)")
        report.append("-" * 40)
        report.append(f"  Simulated PnL:         {trading_metrics['simulated_pnl']:+.2f}%")
        report.append(f"  Sharpe Ratio:          {trading_metrics['sharpe_ratio']:.3f}")
        report.append(f"  Max Drawdown:          {trading_metrics['max_drawdown']:.2f}%")
        report.append(f"  Win Rate:              {trading_metrics['win_rate']:.2f}%")
        report.append(f"  Total Trades:          {trading_metrics['total_trades']}")
        report.append("")

        # Section 5: Hierarchical Level Analysis
        report.append("## 5. HIERARCHICAL LEVEL IMPORTANCE")
        report.append("-" * 40)
        report.append("  How much each level contributed to predictions:")
        report.append("")
        ticker_bar = int(attention_analysis["ticker_weight"] * 20)
        sector_bar = int(attention_analysis["sector_weight"] * 20)
        market_bar = int(attention_analysis["market_weight"] * 20)
        report.append(
            f"  TICKER-LEVEL:     {'#' * ticker_bar:<20} "
            f"{attention_analysis['ticker_weight']*100:.1f}%"
        )
        report.append(
            f"  SECTOR-LEVEL:     {'#' * sector_bar:<20} "
            f"{attention_analysis['sector_weight']*100:.1f}%"
        )
        report.append(
            f"  MARKET-LEVEL:     {'#' * market_bar:<20} "
            f"{attention_analysis['market_weight']*100:.1f}%"
        )
        report.append("")

        # Section 6: Top Features
        report.append("## 6. TOP 10 MOST IMPORTANT FEATURES")
        report.append("-" * 40)
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]
        for i, (feature, importance) in enumerate(sorted_features, 1):
            bar = "#" * int(importance * 50)
            report.append(f"  {i:2d}. {feature:<35} {bar:<25} {importance:.4f}")
        report.append("")

        # Section 7: Sample Predictions
        report.append("## 7. SAMPLE PREDICTIONS")
        report.append("-" * 40)
        for i, sample in enumerate(sample_predictions[:5], 1):
            status = "[OK]" if sample["correct"] else "[X]"
            report.append(f"  Sample {i}: {status}")
            report.append(f"    Ticker:      {sample['ticker']}")
            report.append(f"    Date:        {sample['date']}")
            report.append(f"    Headline:    {sample['headline']}")
            report.append(
                f"    Prediction:  {sample['prediction']} "
                f"(confidence: {sample['confidence']:.2f})"
            )
            report.append(f"    Actual:      {sample['actual']}")
            probs = sample["probs"]
            report.append(
                f"    Probs:       SELL={probs[0]:.2f}, HOLD={probs[1]:.2f}, BUY={probs[2]:.2f}"
            )
            report.append("")

        # Section 8: Calibration Analysis
        report.append("## 8. CALIBRATION ANALYSIS")
        report.append("-" * 40)
        report.append(f"  Brier Score:                    {calibration_metrics['brier_score']:.4f}")
        report.append(f"  Expected Calibration Error:     {calibration_metrics['ece']:.4f}")
        report.append(
            f"  Avg Confidence (Correct):       {calibration_metrics['conf_when_correct']:.4f}"
        )
        report.append(
            f"  Avg Confidence (Incorrect):     {calibration_metrics['conf_when_wrong']:.4f}"
        )
        report.append("")

        # Section 9: AI Summary (if enabled)
        if self.config.summary:
            report.append("## 9. AI-POWERED SUMMARY")
            report.append("-" * 40)
            summary_text = self._generate_ai_summary(
                accuracy=accuracy,
                f1_macro=f1_macro,
                f1_weighted=f1_weighted,
                directional_accuracy=directional_accuracy,
                trading_metrics=trading_metrics,
                attention_analysis=attention_analysis,
                calibration_metrics=calibration_metrics,
                per_class_metrics=per_class_metrics,
                n_samples=n_samples,
            )
            for line in summary_text.split("\n"):
                report.append(f"  {line}")
            report.append("")

        # Footer
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        return report

    def _generate_ai_summary(
        self,
        accuracy: float,
        f1_macro: float,
        f1_weighted: float,
        directional_accuracy: float,
        trading_metrics: Dict[str, float],
        attention_analysis: Dict[str, float],
        calibration_metrics: Dict[str, float],
        per_class_metrics: Dict[str, List],
        n_samples: int,
    ) -> str:
        """
        Generate AI-powered summary of evaluation results.

        Uses the configured summary model (Flan-T5) to analyze results
        and provide recommendations.

        Args:
            accuracy: Overall accuracy
            f1_macro: Macro F1 score
            f1_weighted: Weighted F1 score
            directional_accuracy: Directional accuracy
            trading_metrics: Trading simulation results
            attention_analysis: Level importance weights
            calibration_metrics: Calibration metrics
            per_class_metrics: Per-class precision/recall/f1
            n_samples: Number of test samples

        Returns:
            Summary text string.
        """
        try:
            from src.models.report_summarizer import (
                create_summarizer,
            )

            summary_data = {
                "symbol": self.config.symbol,
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "directional_accuracy": directional_accuracy,
                "n_samples": n_samples,
                "n_train": int(n_samples * self.config.split / (1 - self.config.split)),
                "total_trades": trading_metrics["total_trades"],
                "simulated_pnl": trading_metrics["simulated_pnl"],
                "sharpe_ratio": trading_metrics["sharpe_ratio"],
                "win_rate": trading_metrics["win_rate"],
                "max_drawdown": trading_metrics["max_drawdown"],
                "attention_analysis": attention_analysis,
                "brier_score": calibration_metrics["brier_score"],
                "conf_when_correct": calibration_metrics["conf_when_correct"],
                "conf_when_wrong": calibration_metrics["conf_when_wrong"],
                "per_class_metrics": per_class_metrics,
            }

            summarizer = create_summarizer(
                model_name=self.config.summary_model,
                verbose=False,
            )
            return summarizer.summarize(summary_data)

        except Exception as e:
            return (
                f"Error generating summary: {e}\n"
                "Try installing: pip install transformers sentencepiece"
            )

    def print_report(self) -> None:
        """
        Print the evaluation report to stdout.

        Must be called after run() has been executed.

        Raises:
            RuntimeError: If run() has not been called yet.
        """
        if self.results is None:
            raise RuntimeError("Must call run() before print_report()")

        for line in self.results["report_lines"]:
            print(line)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model configuration and device info.

        Raises:
            RuntimeError: If model has not been loaded.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call run() first.")

        return {
            "model_file": self.config.model_file,
            "model_type": "HierarchicalSentimentTransformer",
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "seq_length": self.config.seq_length,
            "device": str(self.model.device),
        }

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded test data.

        Returns:
            Dictionary with data file path, sample counts, and label distribution.

        Raises:
            RuntimeError: If data has not been loaded.
        """
        if self.test_data is None:
            raise RuntimeError("Data not loaded. Call run() first.")

        test_sequences, test_labels, test_timestamps, news_items = self.test_data

        import numpy as np

        label_counts = {
            "sell": int(np.sum(test_labels == 0)),
            "hold": int(np.sum(test_labels == 1)),
            "buy": int(np.sum(test_labels == 2)),
        }

        return {
            "data_file": self.config.data_file,
            "total_news_items": len(news_items),
            "test_sequences": len(test_sequences),
            "sequence_shape": test_sequences.shape if len(test_sequences) > 0 else None,
            "label_distribution": label_counts,
            "split_ratio": self.config.split,
        }
