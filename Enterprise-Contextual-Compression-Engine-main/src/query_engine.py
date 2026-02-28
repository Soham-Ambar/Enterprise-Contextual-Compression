"""
Query Engine Module

Provides semantic search over compressed facts using sentence embeddings
and cosine similarity. Supports drill-down to original source text.
"""

import json
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Semantic search engine for querying compressed facts.
    """
    
    def __init__(
        self,
        compressed_data_path: Optional[str] = None,
        compressed_data: Optional[Dict[str, Any]] = None,
        model_name: str = 'all-MiniLM-L6-v2',
        top_k: int = 5
    ):
        """
        Initialize the query engine.
        
        Args:
            compressed_data_path: Path to compressed_output.json file
            compressed_data: Optional pre-loaded compressed data dictionary
            model_name: Sentence transformer model name
            top_k: Number of top results to return
        """
        logger.info("Initializing Query Engine...")
        
        # Load compressed data
        if compressed_data is not None:
            self.compressed_data = compressed_data
        elif compressed_data_path:
            self.compressed_data = self._load_compressed_data(compressed_data_path)
        else:
            raise ValueError("Either compressed_data_path or compressed_data must be provided")
        
        # Extract facts
        self.facts = self.compressed_data.get('compressed_facts', [])
        
        if not self.facts:
            raise ValueError("No compressed facts found in data")
        
        logger.info(f"Loaded {len(self.facts)} compressed facts")
        
        # Initialize sentence transformer
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Generate embeddings for all facts
        logger.info("Generating embeddings for facts...")
        self.fact_texts = [fact.get('fact_text', '') for fact in self.facts]
        self.fact_embeddings = self.model.encode(
            self.fact_texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        logger.info("Query engine initialized successfully")
        
        self.top_k = top_k
        self.drilldown_manager = None
    
    def _load_compressed_data(self, file_path: str) -> Dict[str, Any]:
        """
        Load compressed data from JSON file.
        
        Args:
            file_path: Path to compressed_output.json
            
        Returns:
            Loaded compressed data dictionary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded compressed data from: {file_path}")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Compressed data file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in compressed data file: {e}")
    
    def set_drilldown_manager(self, drilldown_manager):
        """
        Set drill-down manager for source retrieval.
        
        Args:
            drilldown_manager: DrillDownManager instance
        """
        self.drilldown_manager = drilldown_manager
        logger.info("Drill-down manager set")
    
    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Query compressed facts using semantic similarity.
        
        Args:
            query_text: User query string
            top_k: Number of top results to return (overrides default)
            
        Returns:
            List of matching facts with similarity scores, sorted by relevance
        """
        if not query_text.strip():
            return []
        
        k = top_k if top_k is not None else self.top_k
        
        # Generate embedding for query
        query_embedding = self.model.encode(
            [query_text],
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, self.fact_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Build results
        results = []
        for idx in top_indices:
            fact = self.facts[idx].copy()
            similarity_score = float(similarities[idx])
            
            result = {
                'fact_text': fact.get('fact_text', ''),
                'importance_score': fact.get('importance_score', 0.0),
                'confidence_score': fact.get('confidence_score', 0.0),
                'combined_score': fact.get('combined_score', 0.0),
                'original_text': fact.get('original_text', None),
                'fact_type': fact.get('fact_type', 'unknown'),
                'similarity_score': round(similarity_score, 4),
                'document_id': fact.get('document_id', 'unknown'),
                'section_id': fact.get('section_id', 'unknown'),
                'paragraph_id': fact.get('paragraph_id', 'unknown'),
                'source': fact.get('source', {})
            }
            
            results.append(result)
        
        logger.info(f"Query returned {len(results)} results")
        return results
    
    def get_source_text(
        self,
        fact: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve original source text for a fact using drill-down.
        
        Args:
            fact: Fact dictionary with source information
            
        Returns:
            Dictionary with source text information or None if not available
        """
        document_id = fact.get('document_id', 'unknown')
        section_id = fact.get('section_id', 'unknown')
        paragraph_id = fact.get('paragraph_id', 'unknown')
        
        # Try drill-down manager first
        if self.drilldown_manager:
            source_info = self.drilldown_manager.get_fact_source(fact)
            
            if source_info:
                return {
                    'paragraph_text': source_info['paragraph'].get('text', ''),
                    'section_title': source_info.get('section_title', 'Unknown Section'),
                    'document_id': document_id
                }

        # Fallback: if fact includes embedded original_text, return that
        if fact.get('original_text'):
            return {
                'paragraph_text': fact.get('original_text'),
                'section_title': 'Unknown Section',
                'document_id': document_id
            }
        
        # Fallback: Try to get from traceability manager if available
        if self.drilldown_manager and hasattr(self.drilldown_manager, 'traceability'):
            traceability = self.drilldown_manager.traceability
            original_text = traceability.get_original_paragraph(
                document_id, section_id, paragraph_id
            )
            
            if original_text:
                return {
                    'paragraph_text': original_text,
                    'section_title': 'Unknown Section',
                    'document_id': document_id
                }
        
        return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get document metadata from compressed data.
        
        Returns:
            Metadata dictionary
        """
        return self.compressed_data.get('metadata', {})
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics.
        
        Returns:
            Compression statistics dictionary
        """
        return self.compressed_data.get('compression_stats', {})
