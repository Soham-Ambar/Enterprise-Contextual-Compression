"""
Main Pipeline Module

Orchestrates the complete contextual compression pipeline.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ingestion import DocumentIngester
from chunking import HierarchicalChunker
from extraction import FactExtractor
from liquid_nn import ImportanceScorer
from traceability import TraceabilityManager
from compression import HierarchicalCompressor
from threshold_tuner import ThresholdTuner
from validation import Validator
from drilldown import DrillDownManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContextualCompressionEngine:
    """
    Main engine that orchestrates the complete compression pipeline.
    """
    
    def __init__(
        self,
        importance_threshold: float = 0.5,
        model_path: Optional[str] = None,
        use_combined_score: bool = True,
        min_confidence: float = 0.4,
        auto_tune_threshold: bool = False,
        target_compression_ratio: float = 0.5
    ):
        """
        Initialize the compression engine.
        
        Args:
            importance_threshold: Minimum importance score for fact retention
            model_path: Optional path to pre-trained LNN model
            use_combined_score: Use combined score (LNN + extraction confidence)
            min_confidence: Minimum confidence score for fact retention
            auto_tune_threshold: Automatically tune threshold for optimal compression
            target_compression_ratio: Target compression ratio for auto-tuning
        """
        logger.info("Initializing Contextual Compression Engine...")
        
        # Initialize components
        self.ingester = DocumentIngester()
        self.chunker = HierarchicalChunker()
        self.extractor = FactExtractor()
        self.scorer = ImportanceScorer(model_path=model_path)
        self.traceability = TraceabilityManager()
        self.tuner = ThresholdTuner()
        self.compressor = HierarchicalCompressor(
            importance_threshold=importance_threshold,
            use_combined_score=use_combined_score,
            min_confidence=min_confidence
        )
        self.drilldown = DrillDownManager(self.traceability)
        
        self.auto_tune_threshold = auto_tune_threshold
        self.target_compression_ratio = target_compression_ratio
        
        logger.info("Engine initialized successfully")
    
    def process_document(
        self,
        file_path: str,
        output_path: Optional[str] = None
    ) -> dict:
        """
        Process a document through the complete compression pipeline.
        
        Args:
            file_path: Path to input document (txt or pdf)
            output_path: Optional path to save output JSON
            
        Returns:
            Dictionary containing compressed facts and metadata
        """
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Step 0: Validate input
            Validator.validate_file_path(file_path)
            
            # Step 1: Ingest document
            logger.info("Step 1: Ingesting document...")
            document = self.ingester.load_document(file_path)
            logger.info(f"Loaded document: {document['document_id']}")
            
            # Step 2: Chunk into hierarchical structure
            logger.info("Step 2: Chunking document into hierarchical structure...")
            structured_document = self.chunker.chunk_document(document)
            logger.info(
                f"Created {len(structured_document['sections'])} sections"
            )
            
            # Register document structure for drill-down
            self.drilldown.register_document_structure(
                structured_document['document_id'],
                structured_document
            )
            
            # Validate document structure
            Validator.validate_document_structure(structured_document)
            
            # Step 3: Extract facts
            logger.info("Step 3: Extracting decision-critical facts...")
            all_facts = []
            for section in structured_document['sections']:
                for paragraph in section['paragraphs']:
                    facts = self.extractor.extract_facts(
                        paragraph,
                        section['section_id'],
                        structured_document['document_id']
                    )
                    all_facts.extend(facts)
            
            logger.info(f"Extracted {len(all_facts)} facts")
            
            # Step 4: Score facts using Liquid Neural Network
            logger.info("Step 4: Scoring facts using Liquid Neural Network...")
            scored_facts = self.scorer.score_facts(all_facts)
            logger.info("Fact scoring completed")
            
            # Step 4.5: Auto-tune threshold if enabled
            if self.auto_tune_threshold and scored_facts:
                logger.info("Step 4.5: Auto-tuning importance threshold...")
                optimal_threshold, tune_stats = self.tuner.find_optimal_threshold(
                    scored_facts,
                    target_compression_ratio=self.target_compression_ratio
                )
                self.compressor.importance_threshold = optimal_threshold
                logger.info(f"Threshold tuned to: {optimal_threshold:.3f}")
            
            # Step 5: Register facts for traceability
            logger.info("Step 5: Registering facts for traceability...")
            for fact in scored_facts:
                self.traceability.register_fact(fact)
            logger.info("Traceability registration completed")
            
            # Step 6: Perform hierarchical compression
            logger.info("Step 6: Performing hierarchical compression...")
            compressed_document = self.compressor.compress_document(
                structured_document,
                scored_facts
            )
            logger.info("Compression completed")
            
            # Step 7: Generate output JSON
            logger.info("Step 7: Generating output JSON...")
            output_json = self.compressor.generate_output_json(
                compressed_document,
                self.traceability
            )
            
            # Validate output
            Validator.validate_output_json(output_json)
            
            # Save output if path provided
            if output_path:
                self._save_output(output_json, output_path)
                logger.info(f"Output saved to: {output_path}")
            
            logger.info("Document processing completed successfully")
            return output_json
            
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            raise
    
    def process_multiple_documents(
        self,
        file_paths: List[str],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process multiple documents and combine results.
        
        Args:
            file_paths: List of document file paths
            output_path: Optional path to save combined output JSON
            
        Returns:
            Dictionary containing compressed facts from all documents
        """
        logger.info(f"Processing {len(file_paths)} documents...")
        
        all_compressed_facts = []
        document_metadata = []
        
        for idx, file_path in enumerate(file_paths, 1):
            logger.info(f"\nProcessing document {idx}/{len(file_paths)}: {file_path}")
            try:
                result = self.process_document(file_path, output_path=None)
                
                # Add document facts to combined list
                facts = result.get('compressed_facts', [])
                all_compressed_facts.extend(facts)
                
                # Store metadata
                document_metadata.append({
                    'document_id': result['metadata']['document_id'],
                    'file_path': file_path,
                    'fact_count': len(facts),
                    'compression_stats': result['metadata']['compression_stats']
                })
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                document_metadata.append({
                    'document_id': f'error_{idx}',
                    'file_path': file_path,
                    'error': str(e)
                })
        
        # Sort all facts by combined score
        sort_key = 'combined_score' if all_compressed_facts and 'combined_score' in all_compressed_facts[0] else 'importance_score'
        all_compressed_facts.sort(
            key=lambda f: f.get(sort_key, 0.0),
            reverse=True
        )
        
        # Create combined output
        combined_output = {
            'compressed_facts': all_compressed_facts,
            'metadata': {
                'total_documents': len(file_paths),
                'processed_documents': len([m for m in document_metadata if 'error' not in m]),
                'total_facts': len(all_compressed_facts),
                'documents': document_metadata
            }
        }
        
        # Save if output path provided
        if output_path:
            self._save_output(combined_output, output_path)
            logger.info(f"Combined output saved to: {output_path}")
        
        return combined_output
    
    def _save_output(self, output_json: dict, output_path: str):
        """
        Save output JSON to file with clean formatting.
        
        Args:
            output_json: Output dictionary
            output_path: Path to save JSON file
        """
        # Ensure directory exists
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                output_json,
                f,
                indent=2,
                ensure_ascii=False,
                sort_keys=False
            )


def main():
    """Main entry point for the compression engine."""
    parser = argparse.ArgumentParser(
        description='Enterprise Contextual Compression Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single document
  python src/main.py document.pdf -o output.json -t 0.6
  
  # Process multiple documents
  python src/main.py doc1.txt doc2.pdf doc3.txt -o combined.json --auto-tune
  
  # Use auto-tuning with target compression ratio
  python src/main.py document.txt --auto-tune --target-ratio 0.4
        """
    )
    parser.add_argument(
        'input_files',
        type=str,
        nargs='+',
        help='Path(s) to input document(s) (txt or pdf). Multiple files supported.'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to output JSON file'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.5,
        help='Importance threshold for fact retention (0.0-1.0, default: 0.5)'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default=None,
        help='Path to pre-trained LNN model weights'
    )
    parser.add_argument(
        '--auto-tune',
        action='store_true',
        help='Automatically tune threshold for optimal compression'
    )
    parser.add_argument(
        '--target-ratio',
        type=float,
        default=0.5,
        help='Target compression ratio for auto-tuning (0.0-1.0, default: 0.5)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.4,
        help='Minimum confidence score for fact retention (0.0-1.0, default: 0.4)'
    )
    parser.add_argument(
        '--use-combined-score',
        action='store_true',
        default=True,
        help='Use combined score (LNN + extraction confidence)'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    input_paths = []
    for input_file in args.input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_file}")
            sys.exit(1)
        input_paths.append(str(input_path))
    
    # Set default output path if not provided
    output_path = args.output
    if output_path is None:
        if len(input_paths) == 1:
            input_path = Path(input_paths[0])
            output_path = str(input_path.parent / f"{input_path.stem}_compressed.json")
        else:
            output_path = "combined_compressed.json"
    
    # Validate thresholds
    if not 0.0 <= args.threshold <= 1.0:
        logger.error("Threshold must be between 0.0 and 1.0")
        sys.exit(1)
    if not 0.0 <= args.target_ratio <= 1.0:
        logger.error("Target ratio must be between 0.0 and 1.0")
        sys.exit(1)
    if not 0.0 <= args.min_confidence <= 1.0:
        logger.error("Min confidence must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Initialize and run engine
    try:
        engine = ContextualCompressionEngine(
            importance_threshold=args.threshold,
            model_path=args.model,
            use_combined_score=args.use_combined_score,
            min_confidence=args.min_confidence,
            auto_tune_threshold=args.auto_tune,
            target_compression_ratio=args.target_ratio
        )
        
        # Process single or multiple documents
        if len(input_paths) == 1:
            result = engine.process_document(
                file_path=input_paths[0],
                output_path=output_path
            )
            
            # Print summary
            print("\n" + "="*60)
            print("COMPRESSION COMPLETE")
            print("="*60)
            print(f"Document: {result['metadata']['document_id']}")
            print(f"Total facts extracted: {result['metadata']['compression_stats']['total_facts']}")
            print(f"Important facts selected: {result['metadata']['compression_stats']['selected_facts']}")
            print(f"Compression ratio: {result['metadata']['compression_stats']['compression_ratio']:.2%}")
            print(f"Output saved to: {output_path}")
            print("="*60 + "\n")
        else:
            result = engine.process_multiple_documents(
                file_paths=input_paths,
                output_path=output_path
            )
            
            # Print summary
            print("\n" + "="*60)
            print("MULTI-DOCUMENT COMPRESSION COMPLETE")
            print("="*60)
            print(f"Total documents: {result['metadata']['total_documents']}")
            print(f"Processed documents: {result['metadata']['processed_documents']}")
            print(f"Total facts: {result['metadata']['total_facts']}")
            print(f"Output saved to: {output_path}")
            print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
