"""
Gemma Transformer Classifier Model

This module defines the neural network architecture for trading signal classification.
It combines Google's Gemma embeddings with a PyTorch Transformer to classify text
(news headlines + price) into BUY/SELL/HOLD signals.

Architecture:
    Input Text → Gemma Embedding (1024 dim) → Linear Projection (256 dim)
    → Transformer Encoder → Classifier → [SELL, HOLD, BUY] probabilities

Usage:
    This class is used by train.py and test.py. You don't run this file directly.

    # In train.py or test.py:
    from models.gemma_transformer_classifier import SimpleGemmaTransformerClassifier
    model = SimpleGemmaTransformerClassifier()
    logits = model(["Price: 95000\nHeadline: Bitcoin rises"])
    probs = logits.softmax(dim=-1)  # [sell_prob, hold_prob, buy_prob]

Reference:
    Gemma Embedding Model: https://huggingface.co/google/embeddinggemma-300m
    yFinancial: https://ranaroussi.github.io/yfinance/reference/index.html
    
"""

import hashlib
from typing import Iterable

import torch
from torch import Tensor, nn
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION - Model Architecture Parameters
# =============================================================================

# Gemma embedding model from HuggingFace
EMBEDDING_MODEL = "google/embeddinggemma-300m"

# Model architecture
NUM_CLASSES = 3          # SELL, HOLD, BUY
HIDDEN_DIM = 256         # Internal dimension after projection
NUM_LAYERS = 2           # Number of Transformer encoder layers
NUM_HEADS = 4            # Number of attention heads
DROPOUT = 0.1            # Dropout rate for regularization

# =============================================================================
# END CONFIGURATION
# =============================================================================


def get_best_device():
    """
    Automatically detect and return the best available compute device.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU

    Returns:
        torch.device: The best available device for computation.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS device")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


class SimpleGemmaTransformerClassifier(nn.Module):
    """
    A text classifier that uses Gemma embeddings and a Transformer encoder.

    This model:
    1. Takes text input (news headline + price info)
    2. Converts text to embeddings using Google's Gemma model
    3. Passes embeddings through a Transformer encoder
    4. Outputs probabilities for 3 classes: SELL, HOLD, BUY

    The model does NOT generate text - it only classifies into categories.

    Architecture Diagram:

        Input: "Price: 95000, Headline: Bitcoin drops on news"
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Gemma Embedding          │
                    │   (text → 1024 numbers)       │
                    │   Pre-trained, NOT updated    │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     Linear Projection         │
                    │      1024 → 256 dims          │
                    │      (trainable)              │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Transformer Encoder         │
                    │   (2 layers, 4 heads)         │
                    │   (trainable)                 │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Classifier Head          │
                    │      256 → 3 classes          │
                    │      + Sigmoid activation     │
                    │      (trainable)              │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    Output: [0.75, 0.15, 0.10]
                            SELL  HOLD  BUY

    Attributes:
        device: The compute device (cuda/mps/cpu)
        embedding_model: Pre-trained Gemma model for text embeddings
        embedding_cache: Cache for computed embeddings (saves computation)
        project: Linear layer to reduce embedding dimensions
        transformer: Transformer encoder for learning patterns
        classifier: Final layer that outputs class probabilities
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
    ) -> None:
        """
        Initialize the classifier model.

        Args:
            num_classes: Number of output classes (default: 3 for SELL/HOLD/BUY)
            hidden_dim: Dimension of hidden layers (default: 256)
            num_layers: Number of Transformer encoder layers (default: 2)
            num_heads: Number of attention heads (default: 4)
            dropout: Dropout rate for regularization (default: 0.1)
        """
        super().__init__()

        # Cache for embeddings - avoids recomputing same text
        self.embedding_cache = {}

        # Detect best available device
        self.device = get_best_device()

        # Load pre-trained Gemma embedding model (frozen - not trained)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=str(self.device))

        # Get embedding dimension from Gemma model
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Projection layer: reduce embedding dimensions if needed
        if embedding_dim != hidden_dim:
            self.project = nn.Linear(embedding_dim, hidden_dim)
        else:
            self.project = nn.Identity()

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )

        # Stack multiple encoder layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classifier: outputs probability for each class
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid()
        )

        # Move entire model to device
        self.to(self.device)

    def embedding(self, text: str) -> Tensor:
        """
        Convert text to a vector embedding using Gemma.

        Uses caching to avoid recomputing embeddings for the same text.

        Args:
            text: Input text string

        Returns:
            Tensor: 1024-dimensional embedding vector
        """
        # Create unique key for this text
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Return cached embedding if available
        if key not in self.embedding_cache:
            # Tokenize the text
            features = self.embedding_model.tokenize([text])
            features = {name: tensor.to(self.device) for name, tensor in features.items()}

            # Get embeddings (no gradient tracking - Gemma is frozen)
            with torch.no_grad():
                outputs = self.embedding_model(features)

            # Extract token embeddings and attention mask
            token_embeddings = outputs["token_embeddings"]
            attention_mask = features["attention_mask"]

            # Mean pooling: average all token embeddings (weighted by attention mask)
            mask = attention_mask.unsqueeze(-1)
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # Cache the result
            self.embedding_cache[key] = pooled.squeeze(0)

        return self.embedding_cache[key]

    def forward(self, texts: Iterable[str]) -> Tensor:
        """
        Process text inputs and return classification logits.

        This is the main "query" method - pass in text, get predictions out.

        Args:
            texts: List of text strings to classify

        Returns:
            Tensor: Shape (batch_size, 3) with logits for [SELL, HOLD, BUY]

        Example:
            >>> model = SimpleGemmaTransformerClassifier()
            >>> logits = model(["Bitcoin crashes on news"])
            >>> probs = logits.softmax(dim=-1)
            >>> print(probs)  # tensor([[0.75, 0.15, 0.10]])
        """
        # Convert each text to embedding
        embeddings = torch.stack([self.embedding(text) for text in texts])

        # Project to hidden dimension and add sequence dimension
        hidden = self.project(embeddings).unsqueeze(1)

        # Pass through Transformer encoder
        encoded = self.transformer(hidden)

        # Remove sequence dimension
        pooled = encoded.squeeze(1)

        # Get class predictions
        return self.classifier(pooled)

    def predict(self, text: str) -> dict:
        """
        Convenience method to get a prediction for a single text input.

        Args:
            text: News headline or text to classify

        Returns:
            dict: Contains 'prediction', 'confidence', 'probabilities', and 'recommendation'

        Example:
            >>> model = SimpleGemmaTransformerClassifier()
            >>> model.load_state_dict(torch.load('model.pth', weights_only=True))
            >>> result = model.predict("Bitcoin surges on ETF approval")
            >>> print(result['recommendation'])
            "BUY - The model predicts the price will RISE. Consider buying."
        """
        self.eval()
        labels = ["SELL", "HOLD", "BUY"]

        # Friendly recommendation messages for each prediction
        recommendations = {
            "SELL": "The model predicts the price will DROP. Consider selling.",
            "HOLD": "The model predicts NO significant price change. Consider holding.",
            "BUY": "The model predicts the price will RISE. Consider buying."
        }

        with torch.no_grad():
            logits = self.forward([text])
            probs = logits.softmax(dim=-1).cpu()[0]

        predicted_idx = torch.argmax(probs).item()
        prediction = labels[predicted_idx]
        confidence = probs[predicted_idx].item()

        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'SELL': probs[0].item(),
                'HOLD': probs[1].item(),
                'BUY': probs[2].item(),
            },
            'recommendation': f"{prediction} ({confidence:.1%} confident) - {recommendations[prediction]}"
        }


# =============================================================================
# Module can be imported but not run directly
# =============================================================================
if __name__ == "__main__":
    print("This module defines the model architecture.")
    print("To train: python train.py")
    print("To test:  python test.py")
    print("\nFor quick prediction example:")
    print("  from models.gemma_transformer_classifier import SimpleGemmaTransformerClassifier")
    print("  model = SimpleGemmaTransformerClassifier()")
    print("  result = model.predict('Bitcoin rises on positive news')")
    print("  print(result)")
