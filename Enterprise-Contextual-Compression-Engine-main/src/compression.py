"""
Compression Module

Performs hierarchical compression:
- Select important facts
- Compress at paragraph level
- Compress at section level
- Compress at document level

Enhanced with threshold tuning and improved filtering.
"""

from typing import List, Dict, Any, Optional
import logging
import re
from loss_analyzer import InformationLossAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalCompressor:
    """
    Performs hierarchical compression of documents with threshold tuning support.
    """
    
    def __init__(
        self,
        importance_threshold: float = 0.5,
        use_combined_score: bool = True,
        min_confidence: float = 0.4
    ):
        """
        Initialize the hierarchical compressor.
        
        Args:
            importance_threshold: Minimum importance score for fact retention
            use_combined_score: If True, use combined_score; else use importance_score
            min_confidence: Minimum confidence score for fact retention
        """
        self.importance_threshold = importance_threshold
        self.use_combined_score = use_combined_score
        self.min_confidence = min_confidence
        self.loss_analyzer = InformationLossAnalyzer()
    
    def compress_document(
        self,
        structured_document: Dict[str, Any],
        all_facts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform hierarchical compression of a document.
        
        Args:
            structured_document: Hierarchically structured document
            all_facts: List of all extracted and scored facts
            
        Returns:
            Compressed document structure with selected facts
        """
        document_id = structured_document['document_id']
        logger.info(f"Compressing document: {document_id}")
        
        # Filter facts by importance threshold and confidence
        score_key = 'combined_score' if self.use_combined_score else 'importance_score'
        
        important_facts = [
            fact for fact in all_facts
            if (fact.get(score_key, fact.get('importance_score', 0.0)) >= self.importance_threshold
                and fact.get('confidence_score', 1.0) >= self.min_confidence)
        ]
        
        logger.info(
            f"Selected {len(important_facts)} important facts "
            f"(threshold: {self.importance_threshold})"
        )
        
        # Analyze information loss
        loss_stats = self.loss_analyzer.analyze_compression(all_facts, important_facts)
        
        # Organize facts by section and paragraph
        facts_by_location = self._organize_facts_by_location(important_facts)
        
        # Compress at section level
        compressed_sections = []
        for section in structured_document['sections']:
            section_id = section['section_id']
            section_facts = facts_by_location.get(section_id, {})
            
            compressed_section = self._compress_section(
                section, section_facts
            )
            compressed_sections.append(compressed_section)
        
        result = {
            'document_id': document_id,
            'compressed_facts': important_facts,
            'sections': compressed_sections,
            'compression_stats': loss_stats
        }
        
        # Propagate metadata if present
        if 'metadata' in structured_document:
            result['metadata'] = structured_document['metadata']
        
        return result
    
    def _organize_facts_by_location(
        self,
        facts: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Organize facts by section and paragraph location.
        
        Args:
            facts: List of fact dictionaries
            
        Returns:
            Nested dictionary: {section_id: {paragraph_id: [facts]}}
        """
        organized = {}
        
        for fact in facts:
            section_id = fact.get('section_id', 'unknown')
            paragraph_id = fact.get('paragraph_id', 'unknown')
            
            if section_id not in organized:
                organized[section_id] = {}
            
            if paragraph_id not in organized[section_id]:
                organized[section_id][paragraph_id] = []
            
            organized[section_id][paragraph_id].append(fact)
        
        return organized
    
    def _compress_section(
        self,
        section: Dict[str, Any],
        section_facts: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Compress a section by selecting important facts.
        
        Args:
            section: Section dictionary
            section_facts: Dictionary mapping paragraph IDs to facts
            
        Returns:
            Compressed section dictionary
        """
        compressed_paragraphs = []
        
        for paragraph in section['paragraphs']:
            paragraph_id = paragraph['paragraph_id']
            paragraph_facts = section_facts.get(paragraph_id, [])
            
            if paragraph_facts:
                # Compress paragraph by selecting top facts
                compressed_para = self._compress_paragraph(
                    paragraph, paragraph_facts
                )
                compressed_paragraphs.append(compressed_para)
        
        return {
            'section_id': section['section_id'],
            'title': section['title'],
            'compressed_paragraphs': compressed_paragraphs,
            'fact_count': sum(
                len(facts) for facts in section_facts.values()
            )
        }
    
    def _compress_paragraph(
        self,
        paragraph: Dict[str, Any],
        paragraph_facts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compress a paragraph by selecting important facts.
        
        Args:
            paragraph: Paragraph dictionary
            paragraph_facts: List of facts from this paragraph
            
        Returns:
            Compressed paragraph dictionary
        """
        # Sort facts by importance score (descending)
        sorted_facts = sorted(
            paragraph_facts,
            key=lambda f: f.get('importance_score', 0.0),
            reverse=True
        )
        
        # Select top facts (limit to prevent over-compression)
        max_facts_per_paragraph = 5
        selected_facts = sorted_facts[:max_facts_per_paragraph]
        
        return {
            'paragraph_id': paragraph['paragraph_id'],
            'original_text': paragraph['text'],
            'selected_facts': selected_facts,
            'fact_count': len(selected_facts)
        }
    
    def generate_output_json(
        self,
        compressed_document: Dict[str, Any],
        traceability_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Generate the final output JSON structure.
        
        Args:
            compressed_document: Compressed document dictionary
            traceability_manager: Optional traceability manager instance
            
        Returns:
            JSON structure with compressed facts and source information
        """
        compressed_facts = compressed_document.get('compressed_facts', [])
        
        # Format facts for output with clean formatting
        formatted_facts = []
        for fact in compressed_facts:
            # Enrich with traceability if manager is provided
            if traceability_manager:
                fact = traceability_manager.enrich_fact_with_trace(fact)
            
            # Sanitize fact text (remove excessive whitespace)
            fact_text = re.sub(r'\s+', ' ', fact.get('fact_text', '')).strip()
            
            # Extract traceability fields (check both direct fields and source dictionary)
            source_dict = fact.get('source', {}) if isinstance(fact.get('source'), dict) else {}
            document_id = fact.get('document_id') or source_dict.get('document_id', 'unknown')
            section_id = fact.get('section_id') or source_dict.get('section_id', 'unknown')
            paragraph_id = fact.get('paragraph_id') or source_dict.get('paragraph_id', 'unknown')
            
            formatted_fact = {
                'fact_text': fact_text,
                'importance_score': round(fact.get('importance_score', 0.0), 4),
                'confidence_score': round(fact.get('confidence_score', 0.0), 4),
                'combined_score': round(fact.get('combined_score', fact.get('importance_score', 0.0)), 4),
                'fact_type': fact.get('fact_type', 'unknown'),
                # Required traceability fields as direct fields (for validator)
                'document_id': document_id,
                'section_id': section_id,
                'paragraph_id': paragraph_id,
                # Also keep in source dictionary for backward compatibility
                'source': {
                    'document_id': document_id,
                    'section_id': section_id,
                    'paragraph_id': paragraph_id
                }
            }
            # Include original paragraph text when available for drill-down
            if fact.get('original_text'):
                formatted_fact['original_text'] = fact.get('original_text')
            
            formatted_facts.append(formatted_fact)
        
        # Sort by combined score (or importance score) descending
        sort_key = 'combined_score' if formatted_facts and 'combined_score' in formatted_facts[0] else 'importance_score'
        formatted_facts.sort(
            key=lambda f: f.get(sort_key, 0.0),
            reverse=True
        )
        
        output = {
            'compressed_facts': formatted_facts,
            'metadata': {
                'document_id': compressed_document.get('document_id', 'unknown'),
                **compressed_document.get('metadata', {})  # Include document metadata
            },
            'compression_stats': compressed_document.get('compression_stats', {})
        }
        
        return output
