"""
Drill-Down Module

Provides capability to retrieve original text from compressed facts.
Enables traceability and source verification.
"""

from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrillDownManager:
    """
    Manages drill-down capability for retrieving original document content.
    """
    
    def __init__(self, traceability_manager: Any):
        """
        Initialize the drill-down manager.
        
        Args:
            traceability_manager: TraceabilityManager instance with stored document structure
        """
        self.traceability = traceability_manager
        self.document_structure: Dict[str, Dict[str, Any]] = {}
    
    def register_document_structure(
        self,
        document_id: str,
        structured_document: Dict[str, Any]
    ):
        """
        Register document structure for drill-down access.
        
        Args:
            document_id: Document identifier
            structured_document: Structured document with sections and paragraphs
        """
        self.document_structure[document_id] = structured_document
        logger.debug(f"Registered document structure for: {document_id}")
    
    def get_paragraph(
        self,
        document_id: str,
        section_id: str,
        paragraph_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve original paragraph text by identifiers.
        
        Args:
            document_id: Document identifier
            section_id: Section identifier
            paragraph_id: Paragraph identifier
            
        Returns:
            Dictionary with paragraph information:
            {
                'paragraph_id': str,
                'text': str,
                'section_id': str,
                'section_title': str
            }
            or None if not found
        """
        if document_id not in self.document_structure:
            logger.warning(f"Document not found: {document_id}")
            return None
        
        document = self.document_structure[document_id]
        
        # Find the section
        for section in document.get('sections', []):
            if section.get('section_id') == section_id:
                # Find the paragraph
                for paragraph in section.get('paragraphs', []):
                    if paragraph.get('paragraph_id') == paragraph_id:
                        return {
                            'paragraph_id': paragraph_id,
                            'text': paragraph.get('text', ''),
                            'section_id': section_id,
                            'section_title': section.get('title', 'Unknown Section')
                        }
        
        logger.warning(
            f"Paragraph not found: {document_id}/{section_id}/{paragraph_id}"
        )
        return None
    
    def get_fact_source(self, fact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve original source text for a fact.
        
        Args:
            fact: Fact dictionary with source information
            
        Returns:
            Dictionary with source information:
            {
                'paragraph': {...},
                'section_title': str,
                'document_id': str
            }
            or None if not found
        """
        document_id = fact.get('document_id') or fact.get('source', {}).get('document_id')
        section_id = fact.get('section_id') or fact.get('source', {}).get('section_id')
        paragraph_id = fact.get('paragraph_id') or fact.get('source', {}).get('paragraph_id')
        
        if not all([document_id, section_id, paragraph_id]):
            logger.warning("Incomplete source information in fact")
            return None
        
        paragraph = self.get_paragraph(document_id, section_id, paragraph_id)
        
        if paragraph is None:
            return None
        
        return {
            'paragraph': paragraph,
            'section_title': paragraph.get('section_title', 'Unknown'),
            'document_id': document_id
        }
    
    def get_section(
        self,
        document_id: str,
        section_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve entire section with all paragraphs.
        
        Args:
            document_id: Document identifier
            section_id: Section identifier
            
        Returns:
            Dictionary with section information:
            {
                'section_id': str,
                'title': str,
                'paragraphs': [...]
            }
            or None if not found
        """
        if document_id not in self.document_structure:
            logger.warning(f"Document not found: {document_id}")
            return None
        
        document = self.document_structure[document_id]
        
        # Find the section
        for section in document.get('sections', []):
            if section.get('section_id') == section_id:
                return {
                    'section_id': section_id,
                    'title': section.get('title', 'Unknown Section'),
                    'paragraphs': section.get('paragraphs', [])
                }
        
        logger.warning(f"Section not found: {document_id}/{section_id}")
        return None
    
    def get_all_documents(self) -> List[str]:
        """
        Get list of all registered document IDs.
        
        Returns:
            List of document identifiers
        """
        return list(self.document_structure.keys())
    
    def clear_documents(self):
        """Clear all registered document structures."""
        self.document_structure.clear()
        logger.info("Cleared all document structures")
