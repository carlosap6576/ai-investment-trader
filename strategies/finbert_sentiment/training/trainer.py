"""
Model Trainer for the Hierarchical Sentiment Trading Signal Classifier.

Encapsulates all training business logic:
- Smart Training Guard (data hash checking, cooldown, sample thresholds)
- Model creation and loading
- Training loop with batching
- Evaluation and metrics
- Model saving with metadata

Uses TrainConfig from strategies.finbert_sentiment.training.config.
"""

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from strategies.finbert_sentiment.constants import DATASETS_DIR, LABEL_NAMES
from strategies.finbert_sentiment.training.config import TrainConfig


@dataclass
class TrainingResult:
    """Results from a training run."""
    accuracy: float
    f1_score: float
    total_samples: int
    train_samples: int
    test_samples: int
    epochs_completed: int
    model_file: str
    metadata_file: str
    skipped: bool = False
    skip_reason: str = ""


class ModelTrainer:
    """
    Orchestrates the full training pipeline for the Hierarchical Sentiment model.

    Handles:
    - Smart Training Guard to prevent overfitting
    - Data loading and preprocessing
    - Model initialization (fresh or continued)
    - Training loop with loss tracking
    - Evaluation on test set
    - Model and metadata persistence

    Example:
        config = TrainConfig(symbol="BTC-USD", epochs=50)
        trainer = ModelTrainer(config)
        result = trainer.run()
        print(f"Accuracy: {result.accuracy:.2%}")
    """

    def __init__(self, config: TrainConfig) -> None:
        """
        Initialize the trainer with configuration.

        Args:
            config: TrainConfig instance with all training parameters
        """
        self.config = config
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._device = None

    # =========================================================================
    # SMART TRAINING GUARD
    # =========================================================================

    def calculate_data_hash(self) -> Optional[str]:
        """
        Calculate SHA256 hash of training data content.

        This fingerprints the data to detect changes between training sessions.

        Returns:
            Hash string or None if data file doesn't exist
        """
        if not os.path.exists(self.config.data_file):
            return None

        with open(self.config.data_file, 'r') as f:
            data = json.load(f)

        # Sort by consistent keys to ensure same data = same hash
        sorted_data = sorted(data, key=lambda x: (x.get('pubDate', ''), x.get('title', '')))

        content_str = json.dumps(sorted_data, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def load_training_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Load training metadata from .training_meta.json file.

        Returns:
            Metadata dict or None if file doesn't exist
        """
        if not os.path.exists(self.config.training_meta_file):
            return None

        try:
            with open(self.config.training_meta_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def save_training_metadata(
        self,
        data_hash: str,
        sample_count: int,
        accuracy: float,
        f1: float
    ) -> None:
        """
        Save training metadata after successful training.

        Args:
            data_hash: SHA256 hash of training data
            sample_count: Total number of samples trained on
            accuracy: Final accuracy on test set
            f1: Final F1 score on test set
        """
        metadata = self.load_training_metadata() or {"training_history": []}

        current = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_hash": data_hash,
            "sample_count": sample_count,
            "epochs": self.config.epochs,
            "accuracy": accuracy,
            "f1_score": f1
        }

        metadata["last_training"] = current

        # Keep last 10 history entries
        metadata["training_history"].insert(0, {
            "timestamp": current["timestamp"],
            "samples": sample_count,
            "accuracy": accuracy
        })
        metadata["training_history"] = metadata["training_history"][:10]

        with open(self.config.training_meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def should_train(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Smart Training Guard: Determine if training should proceed.

        Checks:
        1. Force flag (--force bypasses all checks)
        2. Previous training metadata existence
        3. Data hash change (PRIMARY check)
        4. Cooldown period
        5. Minimum new samples threshold

        Returns:
            Tuple of (should_train, reason, details_dict)
        """
        # Force flag bypasses all checks
        if self.config.force:
            return True, "Force flag set (--force)", {}

        metadata = self.load_training_metadata()

        # No previous training = always train
        if metadata is None or "last_training" not in metadata:
            return True, "No previous training found (first training session)", {}

        last = metadata["last_training"]
        details: Dict[str, Any] = {}

        # CHECK 1: Data hash - has the data changed?
        current_hash = self.calculate_data_hash()
        last_hash = last.get("data_hash")
        details["data_hash_changed"] = current_hash != last_hash

        if current_hash == last_hash:
            return False, "Data unchanged since last training", {
                "last_trained": last.get("timestamp"),
                "sample_count": last.get("sample_count"),
                "data_hash": "unchanged"
            }

        # CHECK 2: Cooldown - has enough time passed?
        last_timestamp = last.get("timestamp")
        if last_timestamp:
            try:
                last_time = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                minutes_since = (now - last_time).total_seconds() / 60
                details["minutes_since_last"] = round(minutes_since, 1)

                if minutes_since < self.config.cooldown:
                    return False, f"Cooldown not met ({minutes_since:.0f}min < {self.config.cooldown}min)", {
                        "last_trained": last_timestamp,
                        "minutes_since": round(minutes_since, 1),
                        "cooldown_required": self.config.cooldown
                    }
            except (ValueError, TypeError):
                pass  # If timestamp parsing fails, skip cooldown check

        # CHECK 3: New samples - are there enough new samples?
        with open(self.config.data_file, 'r') as f:
            current_count = len(json.load(f))
        last_count = last.get("sample_count", 0)
        new_samples = current_count - last_count
        details["new_samples"] = new_samples

        if new_samples < self.config.min_new_samples:
            return False, f"Not enough new samples ({new_samples} < {self.config.min_new_samples})", {
                "current_samples": current_count,
                "last_samples": last_count,
                "new_samples": new_samples,
                "required": self.config.min_new_samples
            }

        # All checks passed
        return True, "New data detected - training approved", {
            "new_samples": new_samples,
            "data_hash": "changed",
            "cooldown": "passed"
        }

    def print_guard_decision(self, should: bool, reason: str, details: Dict[str, Any]) -> None:
        """Print the Smart Guard decision in a user-friendly format."""
        print("\n" + "-" * 60)
        print("SMART TRAINING GUARD")
        print("-" * 60)

        if should:
            print(f"  TRAINING APPROVED")
            print(f"    Reason: {reason}")
            if details:
                for key, value in details.items():
                    print(f"    {key}: {value}")
        else:
            print(f"  TRAINING SKIPPED")
            print(f"    Reason: {reason}")
            if details:
                for key, value in details.items():
                    print(f"    {key}: {value}")
            print()
            print(f"  To force training anyway: python -m cli.finbert_sentiment.train -s {self.config.symbol} --force")

        print("-" * 60)

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def check_data_file(self) -> List[Dict[str, Any]]:
        """
        Check if training data exists and has content.

        Returns:
            List of data items

        Raises:
            SystemExit: If data file is missing or empty
        """
        if not os.path.exists(self.config.data_file):
            print(f"\nERROR: Training data file not found: {self.config.data_file}")
            print(f"\nTo create it, run:")
            print(f"  python -m cli.finbert_sentiment.download -s {self.config.symbol}")
            sys.exit(1)

        with open(self.config.data_file, 'r') as f:
            data = json.load(f)

        if len(data) == 0:
            print(f"\nERROR: Training data file is empty: {self.config.data_file}")
            print(f"\nTo populate it, run:")
            print(f"  python -m cli.finbert_sentiment.download -s {self.config.symbol}")
            sys.exit(1)

        return data

    def load_hierarchical_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data and build hierarchical feature vectors for the transformer model.

        Returns:
            Tuple of (train_sequences, train_labels, test_sequences, test_labels)
            where sequences are numpy arrays of shape (num_samples, seq_length, num_features)
        """
        print(f"  Source:           {self.config.data_file}")
        print(f"  Model type:       hierarchical")

        # Lazy import for --help performance
        try:
            from src.data.schemas import NewsItem
            from src.features.feature_builder import TemporalFeatureBuilder, create_label_function
        except ImportError as e:
            sys.stderr.write(f"\nERROR: Hierarchical model data modules not found: {e}\n")
            sys.stderr.write(f"\nMake sure src/data/ and src/features/ modules exist.\n")
            sys.exit(1)

        data = self.check_data_file()

        # Convert to NewsItem objects
        news_items = []
        for item in data:
            try:
                news_item = NewsItem(
                    headline=item.get('title', ''),
                    summary=item.get('summary', ''),
                    timestamp=datetime.fromisoformat(item['pubDate'].replace('Z', '+00:00')),
                    source=item.get('source', 'unknown'),
                    url=item.get('url', ''),
                    level=item.get('level', 'TICKER'),
                    sentiment_score=item.get('sentiment_score', 0.0),
                    sentiment_label=item.get('sentiment_label', 'neutral'),
                    price=item.get('price', 0.0),
                    future_price=item.get('future_price'),
                    price_change_pct=item.get('percentage', 0.0),
                )
                news_items.append(news_item)
            except (KeyError, ValueError):
                continue  # Skip malformed items

        if len(news_items) == 0:
            print(f"\nERROR: No valid news items found in data.")
            print(f"Make sure the data was downloaded with hierarchical fields.")
            print(f"Re-run: python -m cli.finbert_sentiment.download -s {self.config.symbol}")
            sys.exit(1)

        print(f"  Valid news items: {len(news_items)}")

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

        # Build training data
        sequences, labels, timestamps = feature_builder.build_training_data(
            news_items=news_items,
            label_func=label_func,
        )

        print(f"  Total sequences:  {len(sequences)}")
        print(f"  Sequence shape:   {sequences.shape}")

        # Count label distribution
        label_counts = {"sell": 0, "hold": 0, "buy": 0}
        for label in labels:
            if label == 0:
                label_counts["sell"] += 1
            elif label == 1:
                label_counts["hold"] += 1
            else:
                label_counts["buy"] += 1

        total = len(labels)
        if total > 0:
            print(f"  Label distribution:")
            print(f"    SELL:  {label_counts['sell']:>4}  ({label_counts['sell']/total*100:>5.1f}%)")
            print(f"    HOLD:  {label_counts['hold']:>4}  ({label_counts['hold']/total*100:>5.1f}%)")
            print(f"    BUY:   {label_counts['buy']:>4}  ({label_counts['buy']/total*100:>5.1f}%)")

            hold_pct = label_counts["hold"] / total * 100
            if hold_pct > 90:
                print(f"\n  WARNING: {hold_pct:.0f}% of samples are HOLD!")
                print(f"      Consider adjusting thresholds (currently >{self.config.buy_threshold}% / <{self.config.sell_threshold}%)")

        # Split data
        split_index = int(self.config.split * len(sequences))

        train_sequences = sequences[:split_index]
        train_labels = labels[:split_index]
        test_sequences = sequences[split_index:]
        test_labels = labels[split_index:]

        print(f"  Train/Test split:")
        print(f"    Train set:      {len(train_sequences):>4} sequences ({self.config.split*100:.0f}%)")
        print(f"    Test set:       {len(test_sequences):>4} sequences ({(1-self.config.split)*100:.0f}%)")

        return train_sequences, train_labels, test_sequences, test_labels

    # =========================================================================
    # MODEL MANAGEMENT
    # =========================================================================

    def create_model(self) -> None:
        """
        Initialize HierarchicalSentimentTransformer and optimizer.

        Optionally loads existing weights for continuous learning.
        Sets self._model, self._optimizer, self._criterion, self._device.
        """
        # Lazy import for --help performance
        try:
            import torch
        except ImportError:
            sys.stderr.write(f"\nERROR: Required dependencies are not installed.\n")
            sys.stderr.write(f"\nTo install them, run:\n")
            sys.stderr.write(f"  pip install -r requirements.txt\n\n")
            sys.exit(1)

        try:
            from src.models.transformer import HierarchicalSentimentTransformer
            from src.training.losses import TradingSignalLoss
        except ImportError as e:
            sys.stderr.write(f"\nERROR: Model dependencies not found: {e}\n")
            sys.stderr.write(f"\nMake sure src/models/transformer.py and src/training/losses.py exist.\n")
            sys.exit(1)

        model = HierarchicalSentimentTransformer(
            hidden_dim=self.config.hidden_dim,
            num_temporal_layers=self.config.num_layers,
            sequence_length=self.config.seq_length,
        )
        criterion = TradingSignalLoss()

        # Continuous learning: Load existing weights if available and not fresh start
        loaded_existing = False
        if not self.config.fresh and os.path.exists(self.config.model_file):
            try:
                model.load_state_dict(torch.load(self.config.model_file, weights_only=True))
                loaded_existing = True
            except Exception as e:
                print(f"\n  WARNING: Could not load existing model: {e}")
                print(f"           Starting with fresh weights instead.")

        # Create optimizer
        if self.config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.config.learning_rate)

        # Print model info
        print(f"\n" + "-" * 60)
        print("MODEL INITIALIZED")
        print("-" * 60)
        print(f"  Model type:       HierarchicalSentimentTransformer")
        print(f"  Device:           {model.device}")
        print(f"  Sequence length:  {self.config.seq_length}")
        print(f"  Hidden dimension: {self.config.hidden_dim}")
        print(f"  Transformer layers: {self.config.num_layers}")
        print(f"  Optimizer:        {self.config.optimizer}")
        print(f"  Learning rate:    {self.config.learning_rate}")

        # Show continuous learning status
        if loaded_existing:
            print(f"\n  CONTINUING from existing model!")
            print(f"    Loaded weights from: {self.config.model_file}")
            print(f"    (The student is reading their old notes before learning more)")
        elif self.config.fresh:
            print(f"\n  FRESH START (--fresh flag)")
            print(f"    Training from scratch with random weights")
            print(f"    (The student is starting with a blank notebook)")
        else:
            print(f"\n  NEW MODEL (no existing model found)")
            print(f"    Will save to: {self.config.model_file}")
            print(f"    (First day of class - student has no previous notes)")

        print("-" * 60)

        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._device = model.device

    def save_model(self) -> str:
        """
        Save model weights and architecture metadata to disk.

        Returns:
            Path to the metadata file
        """
        import torch

        # Create directory if needed
        model_dir = os.path.dirname(self.config.model_file)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        # Save model weights
        torch.save(self._model.state_dict(), self.config.model_file)

        # Save architecture metadata for auto-detection during testing
        metadata = {
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'seq_length': self.config.seq_length,
            'symbol': self.config.symbol,
            'buy_threshold': self.config.buy_threshold,
            'sell_threshold': self.config.sell_threshold,
        }
        with open(self.config.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "-" * 60)
        print("MODEL SAVED")
        print("-" * 60)
        print(f"  Model file:       {self.config.model_file}")
        print(f"  Metadata file:    {self.config.metadata_file}")
        print(f"  Hidden dim:       {self.config.hidden_dim}")
        print(f"  Num layers:       {self.config.num_layers}")
        print(f"  Seq length:       {self.config.seq_length}")
        print("-" * 60)

        return self.config.metadata_file

    # =========================================================================
    # TRAINING
    # =========================================================================

    def train_model(
        self,
        train_sequences: np.ndarray,
        train_labels: np.ndarray
    ) -> None:
        """
        Train the hierarchical model on temporal sequences.

        Args:
            train_sequences: Training sequences (num_samples, seq_length, num_features)
            train_labels: Training labels (num_samples,)
        """
        import torch

        print(f"  Epochs:           {self.config.epochs}")
        print(f"  Batch size:       {self.config.batch_size}")
        print(f"  Sequences:        {len(train_sequences)}")
        print()

        item_losses = []

        self._model.train()
        for epoch in range(self.config.epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(len(train_sequences))
            sequences = train_sequences[indices]
            targets = train_labels[indices]

            epoch_loss = 0
            num_batches = max(1, len(sequences) // self.config.batch_size)

            for i in range(num_batches):
                start_idx = i * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, len(sequences))

                batch_seq = torch.from_numpy(sequences[start_idx:end_idx]).float().to(self._device)
                batch_target = torch.from_numpy(targets[start_idx:end_idx]).long().to(self._device)

                self._optimizer.zero_grad()
                logits = self._model(batch_seq)

                loss = self._criterion(logits, batch_target)
                loss.backward()
                self._optimizer.step()

                item_losses.append(loss.item())
                epoch_loss += loss.item()

            # Rolling average loss (last 250 samples)
            avg_loss = sum(item_losses[-250:]) / len(item_losses[-250:])
            print(f"  Epoch {epoch + 1}/{self.config.epochs}: avg_loss={avg_loss:.4f}")

    def evaluate_model(
        self,
        test_sequences: np.ndarray,
        test_labels: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluate hierarchical model on test sequences.

        Args:
            test_sequences: Test sequences (num_samples, seq_length, num_features)
            test_labels: Test labels (num_samples,)

        Returns:
            Tuple of (accuracy, f1_score)
        """
        import torch
        from sklearn.metrics import f1_score

        print(f"  Test sequences:   {len(test_sequences)}")

        correct = 0
        total = 0
        all_predictions = []
        all_actuals = []

        self._model.eval()
        with torch.no_grad():
            for i in range(len(test_sequences)):
                seq = torch.from_numpy(test_sequences[i:i+1]).float().to(self._device)
                target = test_labels[i]

                logits = self._model(seq)
                probs = logits.softmax(dim=-1).cpu()
                predicted = torch.argmax(probs, dim=-1).item()

                all_predictions.append(predicted)
                all_actuals.append(target)

                if predicted == target:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0
        f1 = f1_score(all_actuals, all_predictions, average='weighted', zero_division=0)

        print(f"  Results:")
        print(f"    Accuracy:       {correct}/{total} = {accuracy:.2%}")
        print(f"    F1 Score:       {f1:.4f}")

        return accuracy, f1

    def _setup_class_weights(self, train_labels: np.ndarray) -> None:
        """
        Compute and apply class weights to handle imbalanced data.

        Args:
            train_labels: Training labels to compute weights from
        """
        import torch
        from src.training.losses import TradingSignalLoss, compute_class_weights

        train_labels_tensor = torch.tensor(train_labels) if not isinstance(train_labels, torch.Tensor) else train_labels
        class_weights = compute_class_weights(train_labels_tensor, num_classes=3, method='sqrt')

        print(f"\n  Class weights (sqrt method): SELL={class_weights[0]:.3f}, HOLD={class_weights[1]:.3f}, BUY={class_weights[2]:.3f}")

        # Recreate criterion with class weights
        self._criterion = TradingSignalLoss(
            class_weights=class_weights.to(self._device),
            focal_gamma=2.0,
            directional_penalty=1.5,
        )

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def run(self) -> TrainingResult:
        """
        Execute the full training pipeline.

        Steps:
        1. Print configuration
        2. Check Smart Training Guard
        3. Create/load model
        4. Load and preprocess data
        5. Setup class weights
        6. Train model
        7. Evaluate on test set
        8. Save model and metadata

        Returns:
            TrainingResult with accuracy, f1, and other metrics
        """
        print("\n" + "=" * 60)
        print("  HIERARCHICAL SENTIMENT TRADING SIGNAL CLASSIFIER - TRAINING")
        print("=" * 60)
        print(f"\nCONFIGURATION:")
        self._print_config()
        print("\n" + "=" * 60)

        # Smart Training Guard check
        should, reason, details = self.should_train()
        self.print_guard_decision(should, reason, details)

        if not should:
            print("\nExiting (no training needed).")
            return TrainingResult(
                accuracy=0.0,
                f1_score=0.0,
                total_samples=0,
                train_samples=0,
                test_samples=0,
                epochs_completed=0,
                model_file=self.config.model_file,
                metadata_file=self.config.metadata_file,
                skipped=True,
                skip_reason=reason,
            )

        # Calculate data hash before training (for metadata)
        data_hash = self.calculate_data_hash()

        # Create model
        print("\nINITIALIZING MODEL...")
        self.create_model()

        # Load data
        print("\nLOADING DATA...")
        print("-" * 60)
        train_sequences, train_labels, test_sequences, test_labels = self.load_hierarchical_data()

        # Setup class weights for imbalanced data
        self._setup_class_weights(train_labels)

        # Train
        print("\nTRAINING...")
        print("-" * 60)
        self.train_model(train_sequences, train_labels)

        # Evaluate
        print("\n" + "-" * 60)
        print("EVALUATION")
        print("-" * 60)
        accuracy, f1 = self.evaluate_model(test_sequences, test_labels)

        total_samples = len(train_sequences) + len(test_sequences)

        # Save model
        self.save_model()

        # Save training metadata (for Smart Guard)
        self.save_training_metadata(data_hash, total_samples, accuracy, f1)
        print(f"\n  Training metadata saved to: {self.config.training_meta_file}")

        print("\nDone!")

        return TrainingResult(
            accuracy=accuracy,
            f1_score=f1,
            total_samples=total_samples,
            train_samples=len(train_sequences),
            test_samples=len(test_sequences),
            epochs_completed=self.config.epochs,
            model_file=self.config.model_file,
            metadata_file=self.config.metadata_file,
        )

    def _print_config(self) -> None:
        """Print configuration in a formatted way."""
        config = self.config

        # Determine training mode description
        if config.fresh:
            training_mode = "Fresh start (--fresh)"
        else:
            training_mode = "Continue from existing (default)"

        # Smart Guard status
        if config.force:
            guard_status = "BYPASSED (--force)"
        else:
            guard_status = f"Active (min {config.min_new_samples} new samples, {config.cooldown}min cooldown)"

        print(
            f"\n"
            f"  [Data]\n"
            f"  Symbol:           {config.symbol}\n"
            f"  Data file:        {config.data_file}\n"
            f"\n"
            f"  [Signal Thresholds]\n"
            f"  Buy threshold:    > {config.buy_threshold}% price increase\n"
            f"  Sell threshold:   < {config.sell_threshold}% price decrease\n"
            f"\n"
            f"  [Training Parameters]\n"
            f"  Epochs:           {config.epochs}\n"
            f"  Batch size:       {config.batch_size}\n"
            f"  Learning rate:    {config.learning_rate}\n"
            f"  Optimizer:        {config.optimizer}\n"
            f"  Train/Test split: {config.split*100:.0f}% / {(1-config.split)*100:.0f}%\n"
            f"  Training mode:    {training_mode}\n"
            f"\n"
            f"  [Smart Training Guard]\n"
            f"  Guard status:     {guard_status}\n"
            f"\n"
            f"  [Model Architecture]\n"
            f"  Model type:       HierarchicalSentimentTransformer\n"
            f"  Sequence length:  {config.seq_length}\n"
            f"  Hidden dimension: {config.hidden_dim}\n"
            f"  Transformer layers: {config.num_layers}\n"
            f"\n"
            f"  [Output]\n"
            f"  Model file:       {config.model_file}"
        )
