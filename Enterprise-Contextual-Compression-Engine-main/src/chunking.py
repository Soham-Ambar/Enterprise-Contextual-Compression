"""
Document Chunking Module

Breaks documents into hierarchical structure:
Document → Sections → Paragraphs
"""

import re
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalChunker:
    """
    Breaks documents into hierarchical structure.
    """
    
    def __init__(self):
        """Initialize the hierarchical chunker."""
        # Patterns for detecting section headers
        self.section_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^Chapter\s+\d+[.:]\s*(.+)$',  # Chapter headers
            r'^Section\s+\d+[.:]\s*(.+)$',  # Section headers
            r'^\d+\.\d+\s+(.+)$',  # Numbered sections (e.g., 1.1 Title)
            r'^[A-Z][A-Z\s]{3,}$',  # ALL CAPS headers
            r'^[IVX]+\.\s+(.+)$',  # Roman numerals
        ]
    
    def chunk_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Break document into hierarchical structure.
        
        Args:
            document: Dictionary with 'content' and 'document_id'
            
        Returns:
            Dictionary with hierarchical structure:
            {
                'document_id': str,
                'sections': [
                    {
                        'section_id': str,
                        'title': str,
                        'paragraphs': [
                            {
                                'paragraph_id': str,
                                'text': str
                            }
                        ]
                    }
                ]
            }
        """
        content = document['content']
        document_id = document['document_id']
        
        logger.info(f"Chunking document: {document_id}")
        
        # Split into sections
        sections = self._extract_sections(content)
        
        # Process each section to extract paragraphs
        structured_sections = []
        for idx, section in enumerate(sections):
            section_id = f"{document_id}_section_{idx}"
            paragraphs = self._extract_paragraphs(section['content'])
            
            structured_paragraphs = []
            for p_idx, para_text in enumerate(paragraphs):
                if para_text.strip():  # Only include non-empty paragraphs
                    structured_paragraphs.append({
                        'paragraph_id': f"{section_id}_para_{p_idx}",
                        'text': para_text.strip()
                    })
            
            structured_sections.append({
                'section_id': section_id,
                'title': section['title'],
                'paragraphs': structured_paragraphs
            })
        
        logger.info(
            f"Chunked document into {len(structured_sections)} sections "
            f"with {sum(len(s['paragraphs']) for s in structured_sections)} paragraphs"
        )
        
        result = {
            'document_id': document_id,
            'sections': structured_sections
        }
        
        # Propagate metadata if present
        if 'metadata' in document:
            result['metadata'] = document['metadata']
        
        return result
    
    def _extract_sections(self, content: str) -> List[Dict[str, str]]:
        """
        Extract sections from document content.
        
        Args:
            content: Full document content
            
        Returns:
            List of dictionaries with 'title' and 'content'
        """
        lines = content.split('\n')
        sections = []
        current_section = {'title': 'Introduction', 'content': []}
        
        for line in lines:
            # Check if line matches a section header pattern
            is_header = False
            for pattern in self.section_patterns:
                match = re.match(pattern, line.strip(), re.IGNORECASE)
                if match:
                    # Save previous section if it has content
                    if current_section['content']:
                        sections.append({
                            'title': current_section['title'],
                            'content': '\n'.join(current_section['content'])
                        })
                    
                    # Start new section
                    title = match.group(1) if match.groups() else line.strip()
                    current_section = {'title': title.strip(), 'content': []}
                    is_header = True
                    break
            
            if not is_header:
                current_section['content'].append(line)
        
        # Add the last section
        if current_section['content']:
            sections.append({
                'title': current_section['title'],
                'content': '\n'.join(current_section['content'])
            })
        
        # If no sections were found, treat entire document as one section
        if not sections:
            sections.append({
                'title': 'Main Content',
                'content': content
            })
        
        return sections
    
    def _extract_paragraphs(self, section_content: str) -> List[str]:
        """
        Extract paragraphs from section content.
        
        Args:
            section_content: Content of a section
            
        Returns:
            List of paragraph texts
        """
        # Split by double newlines (standard paragraph separator)
        paragraphs = re.split(r'\n\s*\n', section_content)
        
        # Also split long paragraphs that might be concatenated
        refined_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph is very long, try to split by single newlines
            if len(para) > 1000:
                sub_paras = para.split('\n')
                refined_paragraphs.extend([p.strip() for p in sub_paras if p.strip()])
            else:
                refined_paragraphs.append(para)
        
        return refined_paragraphs
