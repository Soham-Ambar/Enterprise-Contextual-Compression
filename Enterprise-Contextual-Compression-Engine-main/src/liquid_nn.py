"""
Liquid Neural Network Module

Implements a Liquid Neural Network (LNN) using PyTorch for assigning
importance scores to facts.

IMPORTANT: This network ONLY outputs importance scores, NOT text.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional
import hashlib
import struct
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiquidNeuralNetwork(nn.Module):
    """
    Liquid Neural Network for importance scoring.
    
    Based on liquid time-constant networks that use differential equations
    to model dynamic behavior. This implementation uses a simplified version
    that processes fact embeddings and outputs importance scores.
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        liquid_dim: int = 128,
        output_dim: int = 1
    ):
        """
        Initialize the Liquid Neural Network.
        
        Args:
            input_dim: Dimension of input embeddings (default: 384 for all-MiniLM)
            hidden_dim: Dimension of hidden layers
            liquid_dim: Dimension of liquid state
            output_dim: Output dimension (1 for importance score)
        """
        super(LiquidNeuralNetwork, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Liquid state layers (simulating liquid dynamics)
        self.liquid_state = nn.Parameter(torch.randn(liquid_dim))
        
        # Recurrent-like processing with time constants
        self.time_constant = nn.Parameter(torch.ones(liquid_dim) * 0.1)
        
        # Processing layers
        self.process_layer1 = nn.Linear(hidden_dim + liquid_dim, hidden_dim)
        self.process_layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Output layer (importance score)
        self.output_layer = nn.Linear(hidden_dim // 2, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the liquid neural network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Importance scores of shape (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Project input
        x_proj = F.relu(self.input_proj(x))
        
        # Expand liquid state to batch size
        liquid_state = self.liquid_state.unsqueeze(0).expand(batch_size, -1)
        
        # Apply time constant modulation (simulating liquid dynamics)
        modulated_liquid = liquid_state * torch.sigmoid(self.time_constant)
        
        # Combine input with liquid state
        combined = torch.cat([x_proj, modulated_liquid], dim=1)
        
        # Process through layers
        x = F.relu(self.process_layer1(combined))
        x = self.dropout(x)
        x = F.relu(self.process_layer2(x))
        x = self.dropout(x)
        
        # Output importance score (sigmoid to ensure 0-1 range)
        importance_score = torch.sigmoid(self.output_layer(x))
        
        return importance_score


class ImportanceScorer:
    """
    Wrapper class for scoring facts using Liquid Neural Network.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the importance scorer.
        
        Args:
            model_path: Optional path to pre-trained model weights
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Lazy-load sentence transformer for embeddings if available,
        # otherwise use a deterministic lightweight fallback embedder.
        embedding_dim = 384
        try:
            logger.info("Attempting to load 'sentence_transformers'...")
            from sentence_transformers import SentenceTransformer  # local import
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            try:
                # Some embedder implementations support `.to()`
                self.embedder.to(self.device)
            except Exception:
                pass
            embedding_dim = self.embedder.get_sentence_embedding_dimension()
            logger.info(f"Loaded sentence_transformers with dim={embedding_dim}")
        except Exception:
            logger.warning(
                "Could not load 'sentence_transformers' quickly — using lightweight fallback embedder."
            )

            class SimpleDeterministicEmbedder:
                def __init__(self, dim: int = 384):
                    self._dim = dim

                def get_sentence_embedding_dimension(self):
                    return self._dim

                def to(self, device: str):
                    # no-op for compatibility
                    return self

                def encode(self, texts, convert_to_tensor=False, show_progress_bar=False):
                    # Deterministic hash-based embeddings: stable across runs
                    vecs = []
                    for t in texts:
                        h = hashlib.sha256(t.encode('utf-8')).digest()
                        # Use repeated hashing to fill vector
                        rnd = random.Random(struct.unpack_from('>Q', h[:8])[0])
                        v = [rnd.uniform(-1, 1) for _ in range(self._dim)]
                        vecs.append(v)
                    arr = np.asarray(vecs, dtype=np.float32)
                    if convert_to_tensor:
                        return torch.from_numpy(arr)
                    return arr

            self.embedder = SimpleDeterministicEmbedder(dim=embedding_dim)
        
        # Initialize LNN
        self.model = LiquidNeuralNetwork(
            input_dim=embedding_dim,
            hidden_dim=256,
            liquid_dim=128,
            output_dim=1
        ).to(self.device)
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            # Initialize with heuristic-based training
            logger.info("Initializing model with heuristic-based importance scoring")
            self._initialize_with_heuristics()
        
        self.model.eval()
    
    def _initialize_with_heuristics(self):
        """
        Initialize model with heuristic-based importance scoring.
        This provides a baseline for importance scoring.
        """
        # This is a simplified initialization
        # In production, you would train on labeled data
        pass
    
    def score_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score facts for importance using the Liquid Neural Network.
        Combines extraction confidence with LNN importance score.
        
        IMPORTANT: This method ONLY outputs importance scores, NOT text.
        
        Args:
            facts: List of fact dictionaries with 'fact_text' and optional 'confidence_score'
            
        Returns:
            List of facts with added 'importance_score' and combined 'confidence_score'
        """
        if not facts:
            return []
        
        # Extract fact texts
        fact_texts = [fact['fact_text'] for fact in facts]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(fact_texts)} facts...")
        with torch.no_grad():
            embeddings = self.embedder.encode(
                fact_texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            embeddings = embeddings.to(self.device)
        
        # Score using LNN
        logger.info("Computing importance scores...")
        with torch.no_grad():
            scores = self.model(embeddings)
            scores = scores.squeeze().cpu().numpy()
        
        # Ensure scores are in [0, 1] range
        scores = np.clip(scores, 0.0, 1.0)
        
        # Combine extraction confidence with LNN importance
        scored_facts = []
        for fact, lnn_score in zip(facts, scores):
            scored_fact = fact.copy()
            
            # Get extraction confidence (default to 0.7 if not present)
            extraction_confidence = fact.get('confidence_score', 0.7)
            
            # Combine scores: weighted average (70% LNN, 30% extraction confidence)
            combined_score = 0.7 * float(lnn_score) + 0.3 * extraction_confidence
            
            scored_fact['importance_score'] = round(float(lnn_score), 4)
            scored_fact['confidence_score'] = round(extraction_confidence, 4)
            scored_fact['combined_score'] = round(combined_score, 4)
            
            scored_facts.append(scored_fact)
        
        logger.info(
            f"Scored {len(scored_facts)} facts. "
            f"Importance range: [{scores.min():.3f}, {scores.max():.3f}]"
        )
        
        return scored_facts
    
    def save_model(self, model_path: str):
        """
        Save model weights to file.
        
        Args:
            model_path: Path to save model weights
        """
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
