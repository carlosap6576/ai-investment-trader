"""
Hierarchical Sentiment Transformer for trading signal classification.

This module implements the HierarchicalSentimentTransformer which processes
multi-level sentiment features (MARKET, SECTOR, TICKER) with cross-level
attention to predict trading signals.

Architecture:
    Level-Specific Encoders → Cross-Level Attention → Temporal Transformer
    → Trading Signal Classification (SELL, HOLD, BUY)
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


# =============================================================================
# CONFIGURATION - Model Architecture Parameters
# =============================================================================

# Feature dimensions (from HierarchicalFeatureVector)
TICKER_DIM = 12    # 12 ticker-level features
SECTOR_DIM = 10    # 10 sector-level features
MARKET_DIM = 10    # 10 market-level features
CROSS_DIM = 8      # 8 cross-level features
TOTAL_FEATURES = TICKER_DIM + SECTOR_DIM + MARKET_DIM + CROSS_DIM  # 40

# Model architecture defaults
NUM_CLASSES = 3          # SELL, HOLD, BUY
HIDDEN_DIM = 128         # Internal dimension for encoders
TEMPORAL_DIM = 64        # Dimension for temporal encoder
NUM_TEMPORAL_LAYERS = 2  # Number of temporal transformer layers
NUM_HEADS = 4            # Number of attention heads
DROPOUT = 0.1            # Dropout rate for regularization
SEQUENCE_LENGTH = 20     # Number of time steps

# =============================================================================
# END CONFIGURATION
# =============================================================================


def get_best_device() -> torch.device:
    """
    Automatically detect and return the best available compute device.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
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


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.

    Adds position information to the feature embeddings so the model
    can distinguish between different time steps.
    """

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LevelEncoder(nn.Module):
    """
    Encoder for a single level of sentiment features.

    Projects raw features to hidden dimension and applies non-linearity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class CrossLevelAttention(nn.Module):
    """
    Cross-level attention mechanism.

    Computes attention between different levels of sentiment features
    to capture how ticker sentiment relates to sector and market sentiment.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query: (batch, seq_len, hidden_dim) - typically ticker features
            key: (batch, seq_len, hidden_dim) - context features
            value: (batch, seq_len, hidden_dim) - context features

        Returns:
            Tuple of (attended_output, attention_weights)
        """
        # Multi-head attention with residual connection
        attn_output, attn_weights = self.attention(query, key, value)
        output = self.norm(query + self.dropout(attn_output))
        return output, attn_weights


class HierarchicalSentimentTransformer(nn.Module):
    """
    Hierarchical Sentiment Transformer for trading signal classification.

    This model processes multi-level sentiment features (MARKET, SECTOR, TICKER)
    through level-specific encoders, combines them with cross-level attention,
    and uses a temporal transformer to capture time-series patterns.

    Architecture:
        ┌─────────────────────────────────────────────────────────────────┐
        │  Input: Temporal sequence of hierarchical feature vectors       │
        │  Shape: (batch_size, seq_len, 40)                               │
        │         [12 ticker + 10 sector + 10 market + 8 cross]           │
        └─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  LEVEL-SPECIFIC ENCODERS                                        │
        │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
        │  │ Ticker   │ │ Sector   │ │ Market   │ │ Cross    │           │
        │  │ Encoder  │ │ Encoder  │ │ Encoder  │ │ Encoder  │           │
        │  │ 12→128   │ │ 10→128   │ │ 10→128   │ │ 8→128    │           │
        │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
        └───────┼────────────┼────────────┼────────────┼──────────────────┘
                │            │            │            │
                ▼            ▼            ▼            ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  CROSS-LEVEL ATTENTION                                          │
        │  Ticker attends to Sector and Market                            │
        │  Output: Enhanced ticker representation                         │
        └─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  FEATURE FUSION                                                 │
        │  Concatenate all level outputs + Projection to temporal_dim    │
        └─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  TEMPORAL TRANSFORMER                                           │
        │  2-layer Transformer Encoder with positional encoding           │
        │  Captures patterns across time steps                            │
        └─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  CLASSIFICATION HEAD                                            │
        │  Mean pooling → Linear → Softmax → [SELL, HOLD, BUY]           │
        └─────────────────────────────────────────────────────────────────┘

    Attributes:
        device: Compute device (cuda/mps/cpu)
        ticker_encoder: Encoder for ticker-level features
        sector_encoder: Encoder for sector-level features
        market_encoder: Encoder for market-level features
        cross_encoder: Encoder for cross-level features
        ticker_to_sector_attn: Attention from ticker to sector
        ticker_to_market_attn: Attention from ticker to market
        fusion_layer: Combines all level outputs
        pos_encoding: Positional encoding for temporal sequence
        temporal_transformer: Transformer for temporal patterns
        classifier: Final classification head
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = HIDDEN_DIM,
        temporal_dim: int = TEMPORAL_DIM,
        num_temporal_layers: int = NUM_TEMPORAL_LAYERS,
        num_heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
        sequence_length: int = SEQUENCE_LENGTH,
    ):
        """
        Initialize the hierarchical sentiment transformer.

        Args:
            num_classes: Number of output classes (3 for SELL/HOLD/BUY)
            hidden_dim: Dimension for level-specific encoders
            temporal_dim: Dimension for temporal transformer
            num_temporal_layers: Number of temporal transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            sequence_length: Expected sequence length (for positional encoding)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.temporal_dim = temporal_dim
        self.device = get_best_device()

        # Level-specific encoders
        self.ticker_encoder = LevelEncoder(TICKER_DIM, hidden_dim, dropout)
        self.sector_encoder = LevelEncoder(SECTOR_DIM, hidden_dim, dropout)
        self.market_encoder = LevelEncoder(MARKET_DIM, hidden_dim, dropout)
        self.cross_encoder = LevelEncoder(CROSS_DIM, hidden_dim, dropout)

        # Cross-level attention
        self.ticker_to_sector_attn = CrossLevelAttention(hidden_dim, num_heads, dropout)
        self.ticker_to_market_attn = CrossLevelAttention(hidden_dim, num_heads, dropout)

        # Feature fusion: combine all levels
        # After attention: ticker (attended) + sector + market + cross = 4 * hidden_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(4 * hidden_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding for temporal sequence
        self.pos_encoding = PositionalEncoding(temporal_dim, sequence_length + 10, dropout)

        # Temporal transformer
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=num_heads,
            dim_feedforward=temporal_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer,
            num_layers=num_temporal_layers,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, num_classes),
        )

        # Store attention weights for interpretability
        self.attention_weights: Dict[str, Optional[Tensor]] = {
            'ticker_to_sector': None,
            'ticker_to_market': None,
        }

        # Move to device
        self.to(self.device)

    def forward(
        self,
        features: Tensor,
        return_attention: bool = False,
    ) -> Tensor:
        """
        Process hierarchical features and return classification logits.

        Args:
            features: Tensor of shape (batch_size, seq_len, 40)
                     Features are ordered: [ticker(12), sector(10), market(10), cross(8)]
            return_attention: If True, store attention weights for analysis

        Returns:
            Tensor of shape (batch_size, 3) with logits for [SELL, HOLD, BUY]
        """
        # Split features by level
        ticker_feat = features[:, :, :TICKER_DIM]
        sector_feat = features[:, :, TICKER_DIM:TICKER_DIM + SECTOR_DIM]
        market_feat = features[:, :, TICKER_DIM + SECTOR_DIM:TICKER_DIM + SECTOR_DIM + MARKET_DIM]
        cross_feat = features[:, :, TICKER_DIM + SECTOR_DIM + MARKET_DIM:]

        # Encode each level
        ticker_enc = self.ticker_encoder(ticker_feat)  # (batch, seq, hidden)
        sector_enc = self.sector_encoder(sector_feat)
        market_enc = self.market_encoder(market_feat)
        cross_enc = self.cross_encoder(cross_feat)

        # Cross-level attention: ticker attends to sector and market
        ticker_sector, attn_ts = self.ticker_to_sector_attn(ticker_enc, sector_enc, sector_enc)
        ticker_market, attn_tm = self.ticker_to_market_attn(ticker_enc, market_enc, market_enc)

        # Store attention weights if requested
        if return_attention:
            self.attention_weights['ticker_to_sector'] = attn_ts.detach()
            self.attention_weights['ticker_to_market'] = attn_tm.detach()

        # Combine attended ticker with average of both attention outputs
        ticker_attended = (ticker_sector + ticker_market) / 2

        # Fuse all levels: attended ticker + sector + market + cross
        fused = torch.cat([ticker_attended, sector_enc, market_enc, cross_enc], dim=-1)
        fused = self.fusion_layer(fused)  # (batch, seq, temporal_dim)

        # Add positional encoding
        fused = self.pos_encoding(fused)

        # Temporal transformer
        temporal_out = self.temporal_transformer(fused)  # (batch, seq, temporal_dim)

        # Global mean pooling over sequence
        pooled = temporal_out.mean(dim=1)  # (batch, temporal_dim)

        # Classification
        logits = self.classifier(pooled)  # (batch, num_classes)

        return logits

    def predict(self, features: Tensor) -> Dict:
        """
        Convenience method to get prediction with probabilities.

        Args:
            features: Tensor of shape (seq_len, 40) or (1, seq_len, 40)

        Returns:
            Dict with 'prediction', 'confidence', 'probabilities'
        """
        self.eval()
        labels = ["SELL", "HOLD", "BUY"]

        # Ensure batch dimension
        if features.dim() == 2:
            features = features.unsqueeze(0)

        features = features.to(self.device)

        with torch.no_grad():
            logits = self.forward(features, return_attention=True)
            probs = torch.softmax(logits, dim=-1).cpu()[0]

        predicted_idx = int(torch.argmax(probs).item())
        prediction = labels[predicted_idx]
        confidence = float(probs[predicted_idx].item())

        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'SELL': float(probs[0].item()),
                'HOLD': float(probs[1].item()),
                'BUY': float(probs[2].item()),
            },
            'attention_weights': {
                k: v.cpu().numpy() if v is not None else None
                for k, v in self.attention_weights.items()
            }
        }

    def get_level_importance(self) -> Dict[str, float]:
        """
        Estimate level importance based on attention weights.

        Returns:
            Dict with importance scores for each level
        """
        importance = {
            'ticker': 1.0,  # Base importance
            'sector': 0.0,
            'market': 0.0,
        }

        # Use attention weights to estimate importance
        if self.attention_weights['ticker_to_sector'] is not None:
            attn_s = self.attention_weights['ticker_to_sector']
            importance['sector'] = float(attn_s.mean().item())

        if self.attention_weights['ticker_to_market'] is not None:
            attn_m = self.attention_weights['ticker_to_market']
            importance['market'] = float(attn_m.mean().item())

        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance


class HybridHierarchicalModel(nn.Module):
    """
    Hybrid model combining Gemma text embeddings with hierarchical sentiment features.

    This model can use both raw text (via Gemma embeddings) and pre-computed
    hierarchical sentiment features for classification.
    """

    def __init__(
        self,
        use_text: bool = True,
        use_features: bool = True,
        text_dim: int = 1024,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
    ):
        """
        Initialize hybrid model.

        Args:
            use_text: Whether to use text embeddings
            use_features: Whether to use hierarchical features
            text_dim: Dimension of text embeddings (1024 for Gemma)
            num_classes: Number of output classes
            hidden_dim: Internal hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.use_text = use_text
        self.use_features = use_features
        self.device = get_best_device()

        # Calculate combined dimension
        combined_dim = 0

        if use_text:
            self.text_encoder = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            combined_dim += hidden_dim

        if use_features:
            self.feature_encoder = nn.Sequential(
                nn.Linear(TOTAL_FEATURES, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            combined_dim += hidden_dim

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.to(self.device)

    def forward(
        self,
        text_embedding: Optional[Tensor] = None,
        features: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through hybrid model.

        Args:
            text_embedding: Tensor of shape (batch, text_dim) - Gemma embeddings
            features: Tensor of shape (batch, 40) - hierarchical features

        Returns:
            Tensor of shape (batch, num_classes) - classification logits
        """
        parts = []

        if self.use_text and text_embedding is not None:
            text_enc = self.text_encoder(text_embedding)
            parts.append(text_enc)

        if self.use_features and features is not None:
            feat_enc = self.feature_encoder(features)
            parts.append(feat_enc)

        if not parts:
            raise ValueError("At least one of text_embedding or features must be provided")

        # Combine and classify
        combined = torch.cat(parts, dim=-1)
        return self.classifier(combined)


def create_hierarchical_model(
    hidden_dim: int = HIDDEN_DIM,
    temporal_dim: int = TEMPORAL_DIM,
    num_layers: int = NUM_TEMPORAL_LAYERS,
    sequence_length: int = SEQUENCE_LENGTH,
) -> HierarchicalSentimentTransformer:
    """
    Factory function to create a hierarchical sentiment transformer.

    Args:
        hidden_dim: Dimension for level encoders
        temporal_dim: Dimension for temporal transformer
        num_layers: Number of temporal transformer layers
        sequence_length: Expected sequence length

    Returns:
        Configured HierarchicalSentimentTransformer instance
    """
    return HierarchicalSentimentTransformer(
        hidden_dim=hidden_dim,
        temporal_dim=temporal_dim,
        num_temporal_layers=num_layers,
        sequence_length=sequence_length,
    )


# =============================================================================
# Module information
# =============================================================================
if __name__ == "__main__":
    print("Hierarchical Sentiment Transformer")
    print("=" * 50)
    print(f"Feature dimensions:")
    print(f"  Ticker:  {TICKER_DIM}")
    print(f"  Sector:  {SECTOR_DIM}")
    print(f"  Market:  {MARKET_DIM}")
    print(f"  Cross:   {CROSS_DIM}")
    print(f"  Total:   {TOTAL_FEATURES}")
    print()
    print("Usage:")
    print("  from src.models.transformer import HierarchicalSentimentTransformer")
    print("  model = HierarchicalSentimentTransformer()")
    print("  logits = model(features)  # features: (batch, seq_len, 40)")
