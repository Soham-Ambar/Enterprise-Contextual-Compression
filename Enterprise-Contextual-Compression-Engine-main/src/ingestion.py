"""
Document Ingestion Module

Handles loading of documents from various formats (txt, pdf).
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime
import pdfplumber
from PyPDF2 import PdfReader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIngester:
    """
    Handles document ingestion from various file formats.
    """
    
    def __init__(self):
        """Initialize the document ingester."""
        self.supported_formats = {'.txt', '.pdf'}
    
    def load_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load a document from file path with metadata support.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata dictionary with fields:
                - author: Optional string
                - date: Optional string
                - category: Optional string
                - version: Optional string
            
        Returns:
            Dictionary containing document content and metadata:
            {
                'content': str,
                'document_id': str,
                'file_path': str,
                'file_type': str,
                'metadata': {
                    'author': str,
                    'date': str,
                    'category': str,
                    'version': str,
                    'source_file': str
                }
            }
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {self.supported_formats}"
            )
        
        document_id = os.path.splitext(os.path.basename(file_path))[0]
        
        logger.info(f"Loading document: {file_path}")
        
        if file_ext == '.txt':
            content = self._load_txt(file_path)
        elif file_ext == '.pdf':
            content = self._load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_ext}")
        
        # Process metadata
        processed_metadata = self._process_metadata(metadata, file_path)
        
        return {
            'content': content,
            'document_id': document_id,
            'file_path': file_path,
            'file_type': file_ext,
            'metadata': processed_metadata
        }
    
    def _process_metadata(
        self,
        metadata: Optional[Dict[str, Any]],
        file_path: str
    ) -> Dict[str, Any]:
        """
        Process and populate metadata with defaults if not provided.
        
        Args:
            metadata: Optional metadata dictionary
            file_path: Path to source file
            
        Returns:
            Complete metadata dictionary with all required fields
        """
        if metadata is None:
            metadata = {}
        
        # Get current timestamp
        current_timestamp = datetime.now().isoformat()
        
        return {
            'author': metadata.get('author', 'unknown'),
            'date': metadata.get('date', current_timestamp),
            'category': metadata.get('category', 'general'),
            'version': metadata.get('version', '1.0'),
            'source_file': file_path
        }
    
    def _load_txt(self, file_path: str) -> str:
        """
        Load text from a .txt file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Content of the text file as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Loaded {len(content)} characters from text file")
            return content
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            logger.info(f"Loaded {len(content)} characters from text file (latin-1)")
            return content
    
    def _load_pdf(self, file_path: str) -> str:
        """
        Load text from a .pdf file using pdfplumber (primary) and PyPDF2 (fallback).
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content from PDF
        """
        content_parts = []
        
        # Try pdfplumber first (better for structured PDFs)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content_parts.append(page_text)
            logger.info(f"Loaded PDF using pdfplumber: {len(pdf.pages)} pages")
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying PyPDF2")
            # Fallback to PyPDF2
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        content_parts.append(page_text)
                logger.info(f"Loaded PDF using PyPDF2: {len(reader.pages)} pages")
            except Exception as e:
                logger.error(f"Both PDF extraction methods failed: {e}")
                raise ValueError(f"Failed to extract text from PDF: {e}")
        
        content = '\n\n'.join(content_parts)
        logger.info(f"Extracted {len(content)} characters from PDF")
        return content
