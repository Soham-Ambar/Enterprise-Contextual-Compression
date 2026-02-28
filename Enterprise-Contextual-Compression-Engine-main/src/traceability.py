"""
Traceability Module

Maintains full traceability for compressed facts, ensuring every fact
can be traced back to its original source.
"""

from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TraceabilityManager:
    """
    Manages traceability for facts throughout the compression pipeline.
    """
    
    def __init__(self):
        """Initialize the traceability manager."""
        self.trace_map: Dict[str, Dict[str, Any]] = {}
        self.paragraph_store: Dict[str, str] = {}  # Store original paragraph text
    
    def register_fact(
        self,
        fact: Dict[str, Any],
        source_info: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register a fact with its source information.
        
        Args:
            fact: Fact dictionary
            source_info: Optional source information dictionary
            
        Returns:
            Unique trace ID for the fact
        """
        # Generate trace ID
        trace_id = self._generate_trace_id(fact)
        
        # Extract source information from fact if not provided
        if source_info is None:
            source_info = {
                'document_id': fact.get('document_id', 'unknown'),
                'section_id': fact.get('section_id', 'unknown'),
                'paragraph_id': fact.get('paragraph_id', 'unknown')
            }
        
        # Store original paragraph text for drill-down
        paragraph_key = f"{source_info['document_id']}_{source_info['section_id']}_{source_info['paragraph_id']}"
        original_paragraph_text = fact.get('original_text', '')
        if original_paragraph_text and paragraph_key not in self.paragraph_store:
            self.paragraph_store[paragraph_key] = original_paragraph_text
        
        # Store trace information
        self.trace_map[trace_id] = {
            'fact_text': fact.get('fact_text', ''),
            'fact_type': fact.get('fact_type', 'unknown'),
            'importance_score': fact.get('importance_score', 0.0),
            'source': source_info,
            'original_text': original_paragraph_text
        }
        
        return trace_id
    
    def get_trace_info(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve trace information for a fact.
        
        Args:
            trace_id: Trace ID of the fact
            
        Returns:
            Trace information dictionary or None if not found
        """
        return self.trace_map.get(trace_id)
    
    def enrich_fact_with_trace(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a fact dictionary with traceability information.
        
        Args:
            fact: Fact dictionary
            
        Returns:
            Enriched fact dictionary with source information
        """
        enriched = fact.copy()
        
        # Ensure source information is present
        if 'source' not in enriched:
            enriched['source'] = {
                'document_id': fact.get('document_id', 'unknown'),
                'section_id': fact.get('section_id', 'unknown'),
                'paragraph_id': fact.get('paragraph_id', 'unknown')
            }
        
        return enriched
    
    def _generate_trace_id(self, fact: Dict[str, Any]) -> str:
        """
        Generate a unique trace ID for a fact.
        
        Args:
            fact: Fact dictionary
            
        Returns:
            Unique trace ID string
        """
        # Use document, section, paragraph IDs and fact text hash
        doc_id = fact.get('document_id', 'unknown')
        sec_id = fact.get('section_id', 'unknown')
        para_id = fact.get('paragraph_id', 'unknown')
        fact_text = fact.get('fact_text', '')
        
        # Create a simple hash-based ID
        import hashlib
        fact_hash = hashlib.md5(fact_text.encode()).hexdigest()[:8]
        
        trace_id = f"{doc_id}_{sec_id}_{para_id}_{fact_hash}"
        return trace_id
    
    def get_all_traces(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all trace information.
        
        Returns:
            Dictionary mapping trace IDs to trace information
        """
        return self.trace_map.copy()
    
    def get_original_paragraph(
        self,
        document_id: str,
        section_id: str,
        paragraph_id: str
    ) -> Optional[str]:
        """
        Retrieve original paragraph text for drill-down.
        
        Args:
            document_id: Document identifier
            section_id: Section identifier
            paragraph_id: Paragraph identifier
            
        Returns:
            Original paragraph text or None if not found
        """
        paragraph_key = f"{document_id}_{section_id}_{paragraph_id}"
        return self.paragraph_store.get(paragraph_key)
    
    def clear_traces(self):
        """Clear all trace information."""
        self.trace_map.clear()
        self.paragraph_store.clear()
        logger.info("All traces cleared")
