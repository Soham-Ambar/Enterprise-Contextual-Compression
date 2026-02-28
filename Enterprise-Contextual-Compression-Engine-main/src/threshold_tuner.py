"""
Threshold Tuning Module

Provides adaptive threshold tuning capabilities for optimal fact selection.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThresholdTuner:
    """
    Tunes importance threshold for optimal fact selection.
    """
    
    def __init__(self):
        """Initialize the threshold tuner."""
        pass
    
    def find_optimal_threshold(
        self,
        facts: List[Dict[str, Any]],
        target_compression_ratio: float = 0.5,
        min_facts: int = 5,
        max_facts: Optional[int] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Find optimal threshold to achieve target compression ratio.
        
        Args:
            facts: List of scored facts
            target_compression_ratio: Desired ratio of facts to retain (0.0-1.0)
            min_facts: Minimum number of facts to retain
            max_facts: Maximum number of facts to retain (optional)
            
        Returns:
            Tuple of (optimal_threshold, statistics_dict)
        """
        if not facts:
            return 0.5, {'total_facts': 0, 'selected_facts': 0, 'compression_ratio': 0.0}
        
        # Get all importance scores
        scores = np.array([f.get('importance_score', 0.0) for f in facts])
        scores_sorted = np.sort(scores)[::-1]  # Descending order
        
        # Calculate target number of facts
        target_count = max(min_facts, int(len(facts) * target_compression_ratio))
        if max_facts:
            target_count = min(target_count, max_facts)
        
        if target_count >= len(facts):
            threshold = scores_sorted[-1] - 0.01 if len(scores_sorted) > 0 else 0.0
        else:
            threshold = scores_sorted[target_count - 1] if target_count > 0 else scores_sorted[0]
        
        # Calculate statistics
        selected_facts = [f for f in facts if f.get('importance_score', 0.0) >= threshold]
        
        stats = {
            'total_facts': len(facts),
            'selected_facts': len(selected_facts),
            'compression_ratio': len(selected_facts) / len(facts) if facts else 0.0,
            'threshold': threshold,
            'score_range': {
                'min': float(scores.min()),
                'max': float(scores.max()),
                'mean': float(scores.mean()),
                'median': float(np.median(scores))
            }
        }
        
        logger.info(
            f"Optimal threshold: {threshold:.3f} "
            f"(selects {len(selected_facts)}/{len(facts)} facts)"
        )
        
        return threshold, stats
    
    def tune_by_percentile(
        self,
        facts: List[Dict[str, Any]],
        percentile: float = 50.0
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Set threshold based on percentile of importance scores.
        
        Args:
            facts: List of scored facts
            percentile: Percentile to use (0-100)
            
        Returns:
            Tuple of (threshold, statistics_dict)
        """
        if not facts:
            return 0.5, {}
        
        scores = np.array([f.get('importance_score', 0.0) for f in facts])
        threshold = np.percentile(scores, 100 - percentile)
        
        selected_facts = [f for f in facts if f.get('importance_score', 0.0) >= threshold]
        
        stats = {
            'total_facts': len(facts),
            'selected_facts': len(selected_facts),
            'compression_ratio': len(selected_facts) / len(facts),
            'threshold': float(threshold),
            'percentile': percentile
        }
        
        return float(threshold), stats
    
    def analyze_score_distribution(
        self,
        facts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze the distribution of importance scores.
        
        Args:
            facts: List of scored facts
            
        Returns:
            Dictionary with distribution statistics
        """
        if not facts:
            return {}
        
        scores = np.array([f.get('importance_score', 0.0) for f in facts])
        
        return {
            'count': len(scores),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'percentiles': {
                '25th': float(np.percentile(scores, 25)),
                '50th': float(np.percentile(scores, 50)),
                '75th': float(np.percentile(scores, 75)),
                '90th': float(np.percentile(scores, 90)),
                '95th': float(np.percentile(scores, 95))
            }
        }
