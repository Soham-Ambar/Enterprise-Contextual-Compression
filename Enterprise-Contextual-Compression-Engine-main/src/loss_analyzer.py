"""
Information Loss Analyzer Module

Calculates information loss metrics and compression quality scores.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InformationLossAnalyzer:
    """
    Analyzes information loss during compression.
    """
    
    def __init__(self):
        """Initialize the information loss analyzer."""
        pass
    
    def analyze_compression(
        self,
        all_facts: List[Dict[str, Any]],
        retained_facts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze compression and calculate information loss metrics.
        
        Args:
            all_facts: All extracted facts (before compression)
            retained_facts: Facts retained after compression
            
        Returns:
            Dictionary with compression statistics:
            {
                'total_facts': int,
                'retained_facts': int,
                'dropped_facts': int,
                'compression_ratio': float,
                'information_retention_score': float,
                'importance_loss': float,
                'average_importance_retained': float,
                'average_importance_dropped': float
            }
        """
        total_facts = len(all_facts)
        retained_count = len(retained_facts)
        dropped_count = total_facts - retained_count
        
        # Calculate compression ratio
        compression_ratio = retained_count / total_facts if total_facts > 0 else 0.0
        
        # Get importance scores
        all_scores = [
            f.get('importance_score', 0.0) for f in all_facts
        ]
        retained_scores = [
            f.get('importance_score', 0.0) for f in retained_facts
        ]
        dropped_facts = [
            f for f in all_facts
            if f not in retained_facts
        ]
        dropped_scores = [
            f.get('importance_score', 0.0) for f in dropped_facts
        ]
        
        # Calculate averages
        avg_importance_retained = (
            np.mean(retained_scores) if retained_scores else 0.0
        )
        avg_importance_dropped = (
            np.mean(dropped_scores) if dropped_scores else 0.0
        )
        
        # Calculate information retention score
        # Formula: (sum importance of retained facts) / (sum importance of all facts)
        sum_all_importance = sum(all_scores) if all_scores else 0.0
        sum_retained_importance = sum(retained_scores) if retained_scores else 0.0
        
        information_retention_score = (
            sum_retained_importance / sum_all_importance
            if sum_all_importance > 0 else 0.0
        )
        
        # Calculate importance loss
        importance_loss = 1.0 - information_retention_score
        
        stats = {
            'total_facts': total_facts,
            'retained_facts': retained_count,
            'dropped_facts': dropped_count,
            'compression_ratio': round(compression_ratio, 4),
            'information_retention_score': round(information_retention_score, 4),
            'importance_loss': round(importance_loss, 4),
            'average_importance_retained': round(avg_importance_retained, 4),
            'average_importance_dropped': round(avg_importance_dropped, 4)
        }
        
        logger.info(
            f"Compression analysis: {retained_count}/{total_facts} facts retained "
            f"(retention score: {information_retention_score:.3f})"
        )
        
        return stats
    
    def generate_loss_report(
        self,
        stats: Dict[str, Any]
    ) -> str:
        """
        Generate a human-readable loss report.
        
        Args:
            stats: Compression statistics dictionary
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("INFORMATION LOSS ANALYSIS")
        report.append("=" * 60)
        report.append(f"Total Facts Extracted: {stats['total_facts']}")
        report.append(f"Facts Retained: {stats['retained_facts']}")
        report.append(f"Facts Dropped: {stats['dropped_facts']}")
        report.append(f"Compression Ratio: {stats['compression_ratio']:.2%}")
        report.append(f"Information Retention Score: {stats['information_retention_score']:.2%}")
        report.append(f"Importance Loss: {stats['importance_loss']:.2%}")
        report.append(f"Avg Importance (Retained): {stats['average_importance_retained']:.3f}")
        report.append(f"Avg Importance (Dropped): {stats['average_importance_dropped']:.3f}")
        report.append("=" * 60)
        
        return "\n".join(report)
