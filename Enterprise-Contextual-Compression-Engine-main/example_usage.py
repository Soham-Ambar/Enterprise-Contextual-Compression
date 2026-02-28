"""
Example usage of the Contextual Compression Engine.

This script demonstrates how to use the compression engine programmatically.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main import ContextualCompressionEngine
import json

def main():
    """Example usage of the compression engine."""
    
    # Initialize the engine
    engine = ContextualCompressionEngine(
        importance_threshold=0.5  # Adjust threshold as needed
    )
    
    # Process a document
    # Replace 'example_document.txt' with your document path
    input_file = 'example_document.txt'
    
    try:
        result = engine.process_document(
            file_path=input_file,
            output_path='output_compressed.json'
        )
        
        # Print results
        print("\nCompression Results:")
        print("=" * 60)
        print(f"Document ID: {result['metadata']['document_id']}")
        print(f"Total Facts: {result['metadata']['compression_stats']['total_facts']}")
        print(f"Selected Facts: {result['metadata']['compression_stats']['selected_facts']}")
        print(f"Compression Ratio: {result['metadata']['compression_stats']['compression_ratio']:.2%}")
        print("\nTop 5 Most Important Facts:")
        print("-" * 60)
        
        for i, fact in enumerate(result['compressed_facts'][:5], 1):
            print(f"\n{i}. Score: {fact['importance_score']:.3f}")
            print(f"   Type: {fact['fact_type']}")
            print(f"   Fact: {fact['fact']}")
            print(f"   Source: {fact['source']['section_id']}")
        
        print("\n" + "=" * 60)
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        print("Please create a test document or update the file path.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
