"""
Demo Script - Full Pipeline Execution

Demonstrates the complete Contextual Compression Engine pipeline:
1. Document Ingestion
2. Hierarchical Chunking
3. Fact Extraction
4. Liquid Neural Network Scoring
5. Hierarchical Compression
6. Output JSON Generation
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main import ContextualCompressionEngine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def print_separator(char='=', length=80):
    """Print a separator line."""
    print(char * length)

def print_section(title):
    """Print a section title."""
    print_separator()
    print(f"  {title}")
    print_separator()

def main():
    """Run the complete compression pipeline demo."""
    
    print("\n")
    print_separator('=')
    print("  ENTERPRISE CONTEXTUAL COMPRESSION ENGINE - DEMO")
    print_separator('=')
    print("\n")
    
    # Configuration
    input_file = 'sample.txt'
    importance_threshold = 0.5
    
    # Check if sample file exists
    if not Path(input_file).exists():
        print(f"ERROR: Sample file '{input_file}' not found!")
        print("Please ensure sample.txt exists in the current directory.")
        return
    
    try:
        # Initialize the compression engine with enhanced features
        print_section("STEP 1: Initializing Compression Engine")
        engine = ContextualCompressionEngine(
            importance_threshold=importance_threshold,
            use_combined_score=True,
            min_confidence=0.4,
            auto_tune_threshold=False  # Can enable for auto-tuning
        )
        print("✓ Engine initialized successfully")
        print(f"  Importance threshold: {importance_threshold}")
        print(f"  Using combined score: True")
        print(f"  Minimum confidence: 0.4")
        print()
        
        # Process the document
        print_section("STEP 2: Processing Document")
        print(f"Input file: {input_file}")
        print()
        
        result = engine.process_document(
            file_path=input_file,
            output_path='compressed_output.json'
        )
        
        # Display results
        print_section("STEP 3: Compression Results")
        
        metadata = result.get('metadata', {})
        stats = result.get('compression_stats', {})
        
        print(f"Document ID: {metadata.get('document_id', 'unknown')}")
        
        # Display metadata if available
        if 'author' in metadata:
            print(f"Author: {metadata.get('author', 'unknown')}")
        if 'date' in metadata:
            print(f"Date: {metadata.get('date', 'unknown')}")
        if 'category' in metadata:
            print(f"Category: {metadata.get('category', 'unknown')}")
        if 'version' in metadata:
            print(f"Version: {metadata.get('version', 'unknown')}")
        
        print()
        print("Compression Statistics:")
        print(f"  Total Facts Extracted: {stats.get('total_facts', 0)}")
        print(f"  Facts Retained: {stats.get('retained_facts', 0)}")
        print(f"  Facts Dropped: {stats.get('dropped_facts', 0)}")
        print(f"  Compression Ratio: {stats.get('compression_ratio', 0.0):.2%}")
        print(f"  Information Retention Score: {stats.get('information_retention_score', 0.0):.2%}")
        print(f"  Importance Loss: {stats.get('importance_loss', 0.0):.2%}")
        print(f"  Avg Importance (Retained): {stats.get('average_importance_retained', 0.0):.3f}")
        print(f"  Avg Importance (Dropped): {stats.get('average_importance_dropped', 0.0):.3f}")
        print()
        
        # Display top facts
        print_section("STEP 4: Top Important Facts")
        
        compressed_facts = result['compressed_facts']
        
        if not compressed_facts:
            print("No facts met the importance threshold.")
        else:
            # Show top 10 facts
            top_facts = compressed_facts[:10]
            
            for i, fact in enumerate(top_facts, 1):
                print(f"\n{i}. Importance Score: {fact['importance_score']:.3f}")
                print(f"   Type: {fact['fact_type'].upper()}")
                print(f"   Fact: {fact['fact_text']}")
                print(f"   Source: {fact['source']['section_id']}")
                print(f"   Paragraph: {fact['source']['paragraph_id']}")
        
        print()
        
        # Display fact type breakdown
        print_section("STEP 5: Fact Type Breakdown")
        
        fact_types = {}
        for fact in compressed_facts:
            fact_type = fact['fact_type']
            fact_types[fact_type] = fact_types.get(fact_type, 0) + 1
        
        for fact_type, count in sorted(fact_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {fact_type.capitalize()}: {count}")
        
        print()
        
        # Display output file info
        print_section("STEP 6: Output Information")
        print(f"✓ Compressed output saved to: compressed_output.json")
        print(f"✓ Total compressed facts: {len(compressed_facts)}")
        print()
        
        # Show sample JSON structure
        print_section("STEP 7: Sample JSON Output Structure")
        if compressed_facts:
            sample_fact = compressed_facts[0]
            print(json.dumps(sample_fact, indent=2, ensure_ascii=False))
            print()
        
        # Demonstrate drill-down capability
        print_section("STEP 8: Drill-Down Example")
        if compressed_facts:
            sample_fact = compressed_facts[0]
            doc_id = sample_fact.get('document_id', 'unknown')
            sec_id = sample_fact.get('section_id', 'unknown')
            para_id = sample_fact.get('paragraph_id', 'unknown')
            
            # Get original paragraph text
            paragraph = engine.drilldown.get_paragraph(doc_id, sec_id, para_id)
            
            if paragraph:
                print(f"Fact: {sample_fact['fact_text'][:80]}...")
                print(f"\nOriginal Paragraph:")
                print("-" * 60)
                print(paragraph.get('text', 'Not available'))
                print("-" * 60)
                print(f"Section: {paragraph.get('section_title', 'Unknown')}")
            else:
                print("Drill-down information not available for this fact.")
        print()
        
        print_separator('=')
        print("  DEMO COMPLETED SUCCESSFULLY")
        print_separator('=')
        print("\n")
        
    except FileNotFoundError as e:
        print(f"\nERROR: File not found - {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
