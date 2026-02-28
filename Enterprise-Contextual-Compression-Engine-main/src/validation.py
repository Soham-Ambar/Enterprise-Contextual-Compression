"""
Validation Module

Provides input validation and data quality checks for enterprise-grade robustness.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Validator:
    """
    Validates inputs and outputs for the compression engine.
    """
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """
        Validate that a file path exists and is readable.
        
        Args:
            file_path: Path to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        path = Path(file_path)
        
        if not path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if not path.stat().st_size > 0:
            raise ValueError(f"File is empty: {file_path}")
        
        return True
    
    @staticmethod
    def validate_threshold(threshold: float, name: str = "threshold") -> bool:
        """
        Validate that a threshold is in valid range.
        
        Args:
            threshold: Threshold value to validate
            name: Name of the threshold for error messages
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if not isinstance(threshold, (int, float)):
            raise ValueError(f"{name} must be a number")
        
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0, got {threshold}")
        
        return True
    
    @staticmethod
    def validate_fact(fact: Dict[str, Any]) -> bool:
        """
        Validate that a fact dictionary has required fields.
        
        Args:
            fact: Fact dictionary to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        required_fields = ['fact_text', 'fact_type', 'document_id']
        
        for field in required_fields:
            if field not in fact:
                raise ValueError(f"Fact missing required field: {field}")
        
        if not isinstance(fact['fact_text'], str) or len(fact['fact_text'].strip()) == 0:
            raise ValueError("fact_text must be a non-empty string")
        
        if 'importance_score' in fact:
            Validator.validate_threshold(fact['importance_score'], "importance_score")
        
        if 'confidence_score' in fact:
            Validator.validate_threshold(fact['confidence_score'], "confidence_score")
        
        return True
    
    @staticmethod
    def validate_document_structure(document: Dict[str, Any]) -> bool:
        """
        Validate document structure.
        
        Args:
            document: Document dictionary to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if 'document_id' not in document:
            raise ValueError("Document missing 'document_id' field")
        
        if 'sections' not in document:
            raise ValueError("Document missing 'sections' field")
        
        if not isinstance(document['sections'], list):
            raise ValueError("'sections' must be a list")
        
        for section in document['sections']:
            if 'section_id' not in section:
                raise ValueError("Section missing 'section_id' field")
            if 'paragraphs' not in section:
                raise ValueError("Section missing 'paragraphs' field")
        
        return True
    
    @staticmethod
    def sanitize_fact_text(text: str) -> str:
        """
        Sanitize fact text by removing excessive whitespace.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def validate_output_json(output: Dict[str, Any]) -> bool:
        """
        Validate output JSON structure.
        
        Args:
            output: Output dictionary to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if 'compressed_facts' not in output:
            raise ValueError("Output missing 'compressed_facts' field")
        
        if 'metadata' not in output:
            raise ValueError("Output missing 'metadata' field")
        
        if not isinstance(output['compressed_facts'], list):
            raise ValueError("'compressed_facts' must be a list")
        
        # Validate each fact
        for fact in output['compressed_facts']:
            Validator.validate_fact(fact)
        
        return True
