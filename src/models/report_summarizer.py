"""
AI-powered report summarizer using Google Flan-T5.

Generates human-readable summaries of trading model evaluation reports
using instruction-tuned language models from HuggingFace.

Supported Models:
- google/flan-t5-small  (80M params)  - Fastest, basic quality
- google/flan-t5-base   (250M params) - Good balance
- google/flan-t5-large  (780M params) - Very good quality
- google/flan-t5-xl     (3B params)   - Best quality (default), needs ~12GB VRAM
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import torch


@dataclass
class SummaryConfig:
    """Configuration for report summarizer."""
    model_name: str = "google/flan-t5-xl"
    max_new_tokens: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    device: Optional[str] = None


class ReportSummarizer:
    """
    Summarizes evaluation reports using instruction-tuned LLM.

    Uses Google's Flan-T5 model for generating natural language summaries
    of trading model performance metrics. Supports multiple model sizes
    for different quality/speed tradeoffs.

    Example:
        >>> summarizer = ReportSummarizer()
        >>> summary = summarizer.summarize({
        ...     'symbol': 'AAPL',
        ...     'accuracy': 0.75,
        ...     'f1_weighted': 0.79,
        ...     ...
        ... })
        >>> print(summary)
    """

    SUPPORTED_MODELS = {
        "google/flan-t5-small": {
            "params": "80M",
            "vram": "~0.5GB",
            "quality": "Basic",
            "speed": "Fastest",
        },
        "google/flan-t5-base": {
            "params": "250M",
            "vram": "~1GB",
            "quality": "Good",
            "speed": "Fast",
        },
        "google/flan-t5-large": {
            "params": "780M",
            "vram": "~3GB",
            "quality": "Very Good",
            "speed": "Medium",
        },
        "google/flan-t5-xl": {
            "params": "3B",
            "vram": "~12GB",
            "quality": "Excellent",
            "speed": "Slow",
        },
    }

    def __init__(
        self,
        model_name: str = "google/flan-t5-xl",
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        verbose: bool = True,
    ):
        """
        Initialize summarizer with specified model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_new_tokens: Maximum tokens to generate
            verbose: Print loading messages
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose

        # Validate model name
        if model_name not in self.SUPPORTED_MODELS:
            supported = ", ".join(self.SUPPORTED_MODELS.keys())
            raise ValueError(
                f"Unsupported model: {model_name}\n"
                f"Supported models: {supported}"
            )

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Lazy loading - model loaded on first use
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load model and tokenizer (lazy loading)."""
        if self._model is not None:
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_info = self.SUPPORTED_MODELS[self.model_name]

        if self.verbose:
            print(f"\n  Loading summary model: {self.model_name}")
            print(f"    Parameters: {model_info['params']}")
            print(f"    VRAM: {model_info['vram']}")
            print(f"    Device: {self.device}")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._model.eval()

            if self.verbose:
                print(f"    Status: Loaded successfully")

        except Exception as e:
            # Fallback to CPU if GPU fails
            if self.device != "cpu":
                if self.verbose:
                    print(f"    Warning: Failed to load on {self.device}, falling back to CPU")
                self.device = "cpu"
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                ).to(self.device)
                self._model.eval()
            else:
                raise RuntimeError(f"Failed to load model: {e}")

    def summarize(self, report_data: Dict[str, Any]) -> str:
        """
        Generate human-readable summary from evaluation metrics.

        Args:
            report_data: Dictionary containing all evaluation metrics

        Returns:
            Formatted summary string with insights and recommendations
        """
        # Ensure model is loaded
        self._load_model()

        # Build prompt from data
        prompt = self._build_prompt(report_data)

        # Generate summary
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1,
            )

        raw_summary = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Format and enhance the summary
        formatted_summary = self._format_summary(raw_summary, report_data)

        return formatted_summary

    def _build_prompt(self, data: Dict[str, Any]) -> str:
        """
        Construct prompt for the model with all metrics.

        The prompt is carefully structured to guide Flan-T5 to produce
        actionable, well-organized summaries.
        """
        # Extract metrics with safe defaults
        symbol = data.get('symbol', 'UNKNOWN')
        accuracy = data.get('accuracy', 0) * 100
        f1_macro = data.get('f1_macro', 0)
        f1_weighted = data.get('f1_weighted', 0)
        directional_accuracy = data.get('directional_accuracy', 0) * 100

        n_samples = data.get('n_samples', 0)
        n_train = data.get('n_train', 0)

        total_trades = data.get('total_trades', 0)
        simulated_pnl = data.get('simulated_pnl', 0)
        sharpe_ratio = data.get('sharpe_ratio', 0)
        win_rate = data.get('win_rate', 0)
        max_drawdown = data.get('max_drawdown', 0)

        # Attention/level weights
        attention = data.get('attention_analysis', {})
        ticker_weight = attention.get('ticker_weight', 0) * 100
        sector_weight = attention.get('sector_weight', 0) * 100
        market_weight = attention.get('market_weight', 0) * 100

        # Calibration
        brier_score = data.get('brier_score', 0)
        conf_correct = data.get('conf_when_correct', 0) * 100
        conf_wrong = data.get('conf_when_wrong', 0) * 100

        # Per-class metrics
        per_class = data.get('per_class_metrics', {})

        prompt = f"""Summarize this trading model evaluation for {symbol}:

ACCURACY: {accuracy:.1f}%
F1 SCORE: {f1_weighted:.3f}
DIRECTIONAL ACCURACY: {directional_accuracy:.1f}%

TRADING RESULTS:
- Trades: {total_trades}
- PnL: {simulated_pnl:+.2f}%
- Sharpe: {sharpe_ratio:.2f}
- Win Rate: {win_rate:.1f}%
- Max Drawdown: {max_drawdown:.2f}%

DATA: {n_samples} test samples, {n_train} training samples

LEVEL IMPORTANCE: Ticker={ticker_weight:.0f}%, Sector={sector_weight:.0f}%, Market={market_weight:.0f}%

CALIBRATION: Brier={brier_score:.3f}, Confidence when correct={conf_correct:.0f}%, wrong={conf_wrong:.0f}%

Write a 3-paragraph summary:
1. Performance assessment (accuracy, F1, what the model does well/poorly)
2. Trading viability (PnL, win rate, is it ready for real trading?)
3. Recommendation (what to improve, final verdict)

Be specific about numbers. End with READY or NOT READY for trading."""

        return prompt

    def _format_summary(self, raw_summary: str, data: Dict[str, Any]) -> str:
        """
        Format and enhance the raw model output.

        Adds structure, emojis, and ensures all key points are covered.
        """
        symbol = data.get('symbol', 'UNKNOWN')
        accuracy = data.get('accuracy', 0) * 100
        f1_weighted = data.get('f1_weighted', 0)
        simulated_pnl = data.get('simulated_pnl', 0)
        total_trades = data.get('total_trades', 0)
        n_samples = data.get('n_samples', 0)
        n_train = data.get('n_train', 0)
        win_rate = data.get('win_rate', 0)

        # Determine readiness and confidence
        readiness, confidence = self._assess_readiness(data)

        # Build structured summary
        lines = []

        # Header with key stats
        lines.append(f"MODEL: {symbol} Hierarchical Sentiment Classifier")
        lines.append("")

        # Performance section
        lines.append("PERFORMANCE OVERVIEW:")
        if accuracy >= 70:
            lines.append(f"  The model achieved {accuracy:.1f}% accuracy with F1={f1_weighted:.3f},")
            lines.append(f"  indicating solid pattern recognition on {n_samples} test samples.")
        elif accuracy >= 50:
            lines.append(f"  The model achieved {accuracy:.1f}% accuracy with F1={f1_weighted:.3f},")
            lines.append(f"  showing moderate performance on {n_samples} test samples.")
        else:
            lines.append(f"  The model achieved only {accuracy:.1f}% accuracy with F1={f1_weighted:.3f},")
            lines.append(f"  indicating poor pattern recognition. More data needed.")
        lines.append("")

        # Trading section
        lines.append("TRADING ASSESSMENT:")
        if total_trades == 0:
            lines.append("  No trades executed - model predicted HOLD for all samples.")
            lines.append("  This suggests the model needs more training to make")
            lines.append("  confident BUY/SELL predictions.")
        elif simulated_pnl > 0:
            lines.append(f"  Executed {total_trades} trades with {simulated_pnl:+.2f}% PnL.")
            lines.append(f"  Win rate of {win_rate:.1f}% shows profitable signal detection.")
        else:
            lines.append(f"  Executed {total_trades} trades with {simulated_pnl:+.2f}% PnL.")
            lines.append(f"  Win rate of {win_rate:.1f}% indicates prediction timing issues.")
        lines.append("")

        # Include model's summary if it's meaningful
        if raw_summary and len(raw_summary) > 50:
            # Clean up the raw summary
            clean_summary = raw_summary.strip()
            if not clean_summary.endswith('.'):
                clean_summary += '.'
            lines.append("AI ANALYSIS:")
            # Word wrap the summary
            words = clean_summary.split()
            current_line = "  "
            for word in words:
                if len(current_line) + len(word) + 1 > 70:
                    lines.append(current_line)
                    current_line = "  " + word
                else:
                    current_line += " " + word if current_line.strip() else "  " + word
            if current_line.strip():
                lines.append(current_line)
            lines.append("")

        # Recommendations
        lines.append("RECOMMENDATIONS:")
        recommendations = self._get_recommendations(data)
        for rec in recommendations:
            lines.append(f"  - {rec}")
        lines.append("")

        # Final verdict
        lines.append("-" * 50)
        if readiness == "READY":
            lines.append(f"VERDICT: READY for paper trading")
        elif readiness == "CAUTIOUS":
            lines.append(f"VERDICT: USE WITH CAUTION - limited confidence")
        else:
            lines.append(f"VERDICT: NOT READY for trading")
        lines.append(f"CONFIDENCE: {confidence}")
        lines.append("-" * 50)

        return "\n".join(lines)

    def _assess_readiness(self, data: Dict[str, Any]) -> tuple:
        """
        Determine trading readiness and confidence level.

        Returns:
            Tuple of (readiness, confidence) strings
        """
        accuracy = data.get('accuracy', 0)
        f1_weighted = data.get('f1_weighted', 0)
        simulated_pnl = data.get('simulated_pnl', 0)
        total_trades = data.get('total_trades', 0)
        n_train = data.get('n_train', 0)
        win_rate = data.get('win_rate', 0)

        # Scoring system
        score = 0

        # Accuracy check
        if accuracy >= 0.75:
            score += 2
        elif accuracy >= 0.60:
            score += 1

        # F1 check
        if f1_weighted >= 0.70:
            score += 2
        elif f1_weighted >= 0.50:
            score += 1

        # Trading performance
        if simulated_pnl > 0:
            score += 2
        elif total_trades > 0:
            score += 1

        # Win rate
        if win_rate >= 50:
            score += 2
        elif win_rate >= 30:
            score += 1

        # Data quantity
        if n_train >= 500:
            score += 2
        elif n_train >= 100:
            score += 1

        # Determine readiness
        if score >= 8:
            readiness = "READY"
            confidence = "HIGH"
        elif score >= 5:
            readiness = "CAUTIOUS"
            confidence = "MEDIUM"
        else:
            readiness = "NOT READY"
            confidence = "LOW"

        # Override for critical failures
        if total_trades == 0:
            readiness = "NOT READY"
            confidence = "LOW - No trades executed"

        if n_train < 50:
            confidence = "VERY LOW - Insufficient training data"

        return readiness, confidence

    def _get_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on metrics."""
        recommendations = []

        n_train = data.get('n_train', 0)
        accuracy = data.get('accuracy', 0)
        total_trades = data.get('total_trades', 0)
        simulated_pnl = data.get('simulated_pnl', 0)
        win_rate = data.get('win_rate', 0)

        attention = data.get('attention_analysis', {})
        ticker_weight = attention.get('ticker_weight', 0)
        sector_weight = attention.get('sector_weight', 0)
        market_weight = attention.get('market_weight', 0)

        # Data recommendations
        if n_train < 100:
            recommendations.append(f"Get more training data (current: {n_train}, target: 500+)")
        elif n_train < 500:
            recommendations.append(f"Consider more training data for reliability ({n_train} samples)")

        # Model recommendations
        if accuracy < 0.60:
            recommendations.append("Increase training epochs or adjust learning rate")

        if total_trades == 0:
            recommendations.append("Model too conservative - adjust class weights or thresholds")
        elif win_rate < 40 and total_trades > 0:
            recommendations.append("Improve signal timing - consider different thresholds")

        # Level utilization
        if ticker_weight > 0.9 and sector_weight < 0.05 and market_weight < 0.05:
            recommendations.append("Model ignores sector/market signals - check feature engineering")

        # Trading recommendations
        if simulated_pnl < 0 and total_trades > 0:
            recommendations.append("Negative PnL - do not use for live trading yet")

        if not recommendations:
            recommendations.append("Model shows good fundamentals - continue monitoring")

        return recommendations

    @classmethod
    def list_models(cls) -> str:
        """Return formatted string of supported models."""
        lines = ["Supported Summary Models:"]
        lines.append("-" * 60)
        for name, info in cls.SUPPORTED_MODELS.items():
            lines.append(f"  {name}")
            lines.append(f"    Params: {info['params']}, VRAM: {info['vram']}")
            lines.append(f"    Quality: {info['quality']}, Speed: {info['speed']}")
        return "\n".join(lines)


def create_summarizer(
    model_name: str = "google/flan-t5-xl",
    device: Optional[str] = None,
    verbose: bool = True,
) -> ReportSummarizer:
    """
    Factory function to create a report summarizer.

    Args:
        model_name: HuggingFace model identifier
        device: Device to use ('cuda', 'cpu', or None for auto)
        verbose: Print loading messages

    Returns:
        Configured ReportSummarizer instance
    """
    return ReportSummarizer(
        model_name=model_name,
        device=device,
        verbose=verbose,
    )
