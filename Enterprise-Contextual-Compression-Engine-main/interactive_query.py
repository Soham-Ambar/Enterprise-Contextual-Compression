"""
Interactive Query Interface

Provides an interactive CLI for querying compressed facts with drill-down capability.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from query_engine import QueryEngine
from drilldown import DrillDownManager
from traceability import TraceabilityManager
from ingestion import DocumentIngester
from chunking import HierarchicalChunker
import json
import logging

# Configure logging (reduce verbosity for interactive use)
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)

def print_separator(char='-', length=80):
    """Print a separator line."""
    print(char * length)

def print_header(text):
    """Print a formatted header."""
    print_separator('=')
    print(f"  {text}")
    print_separator('=')

def display_fact(fact: dict, index: int = None):
    """
    Display a fact in a formatted way.
    
    Args:
        fact: Fact dictionary
        index: Optional index number
    """
    prefix = f"{index}. " if index is not None else ""
    
    print(f"\n{prefix}Fact: {fact['fact_text']}")
    print(f"   Type: {fact['fact_type'].upper()}")
    print(f"   Similarity Score: {fact['similarity_score']:.3f}")
    print(f"   Importance Score: {fact['importance_score']:.3f}")
    print(f"   Confidence Score: {fact['confidence_score']:.3f}")
    print(f"   Source: {fact.get('section_id', 'unknown')}")

def display_source(source_info: dict):
    """
    Display original source text with full context.
    
    Args:
        source_info: Source information dictionary
    """
    print_separator('=')
    print("ORIGINAL SOURCE TEXT")
    print_separator('=')
    print(f"Section: {source_info.get('section_title', 'Unknown')}")
    print(f"Document: {source_info.get('document_id', 'unknown')}")
    print()
    print("Original Paragraph:")
    print("-" * 60)
    paragraph_text = source_info.get('paragraph_text', 'Not available')
    if paragraph_text:
        print(paragraph_text)
    else:
        print("Source text not available.")
    print("-" * 60)

def main():
    """Main interactive query interface."""
    
    print_header("CONTEXTUAL COMPRESSION - INTERACTIVE QUERY ENGINE")
    print()
    
    # Configuration
    compressed_data_path = 'compressed_output.json'
    
    # Check if compressed data exists
    if not Path(compressed_data_path).exists():
        print(f"ERROR: Compressed data file '{compressed_data_path}' not found!")
        print("\nPlease run the compression engine first:")
        print("  python demo.py")
        print("  or")
        print("  python src/main.py sample.txt -o compressed_output.json")
        sys.exit(1)
    
    try:
        # Initialize query engine
        print("Loading query engine...")
        engine = QueryEngine(compressed_data_path=compressed_data_path, top_k=5)
        print("✓ Query engine loaded successfully")
        
        # Get metadata to find source file
        metadata = engine.get_metadata()
        source_file = metadata.get('source_file')
        
        # Set up drill-down manager with document structure
        drilldown_manager = None
        if source_file and Path(source_file).exists():
            try:
                print(f"\nLoading original document for drill-down: {source_file}")
                
                # Initialize components
                ingester = DocumentIngester()
                chunker = HierarchicalChunker()
                traceability = TraceabilityManager()
                
                # Load and chunk original document
                document = ingester.load_document(source_file)
                structured_document = chunker.chunk_document(document)
                
                # Initialize drill-down manager
                drilldown_manager = DrillDownManager(traceability)
                
                # Register document structure
                document_id = structured_document.get('document_id', metadata.get('document_id', 'unknown'))
                drilldown_manager.register_document_structure(document_id, structured_document)
                
                # Set drill-down manager in query engine
                engine.set_drilldown_manager(drilldown_manager)
                
                print("✓ Drill-down capability enabled")
                print(f"  Document structure loaded: {len(structured_document.get('sections', []))} sections")
            except Exception as e:
                print(f"⚠ Warning: Could not load document structure: {e}")
                print("  Drill-down will have limited functionality")
                # Still create drill-down manager even if document loading fails
                traceability = TraceabilityManager()
                drilldown_manager = DrillDownManager(traceability)
                engine.set_drilldown_manager(drilldown_manager)
        else:
            # Try to find source file from document_id
            document_id = metadata.get('document_id', 'unknown')
            possible_files = [
                f"{document_id}.txt",
                f"{document_id}.pdf",
                "sample.txt",  # Default fallback
            ]
            
            source_file = None
            for file_path in possible_files:
                if Path(file_path).exists():
                    source_file = file_path
                    break
            
            if source_file:
                try:
                    print(f"\nLoading original document for drill-down: {source_file}")
                    
                    ingester = DocumentIngester()
                    chunker = HierarchicalChunker()
                    traceability = TraceabilityManager()
                    
                    document = ingester.load_document(source_file)
                    structured_document = chunker.chunk_document(document)
                    
                    drilldown_manager = DrillDownManager(traceability)
                    document_id = structured_document.get('document_id', metadata.get('document_id', 'unknown'))
                    drilldown_manager.register_document_structure(document_id, structured_document)
                    
                    engine.set_drilldown_manager(drilldown_manager)
                    print("✓ Drill-down capability enabled")
                    print(f"  Document structure loaded: {len(structured_document.get('sections', []))} sections")
                except Exception as e:
                    print(f"⚠ Warning: Could not load document structure: {e}")
                    traceability = TraceabilityManager()
                    drilldown_manager = DrillDownManager(traceability)
                    engine.set_drilldown_manager(drilldown_manager)
            else:
                print("⚠ Source file not found. Drill-down will have limited functionality.")
                print("  To enable full drill-down, ensure the original document is available.")
                traceability = TraceabilityManager()
                drilldown_manager = DrillDownManager(traceability)
                engine.set_drilldown_manager(drilldown_manager)
        
        # Display metadata (already loaded above)
        stats = engine.get_compression_stats()
        
        print()
        print_separator()
        print("DOCUMENT INFORMATION")
        print_separator()
        print(f"Document ID: {metadata.get('document_id', 'unknown')}")
        if 'author' in metadata:
            print(f"Author: {metadata.get('author', 'unknown')}")
        if 'category' in metadata:
            print(f"Category: {metadata.get('category', 'unknown')}")
        print(f"Total Facts: {stats.get('total_facts', 0)}")
        print(f"Retained Facts: {stats.get('retained_facts', 0)}")
        print(f"Information Retention: {stats.get('information_retention_score', 0.0):.2%}")
        print()
        
        # Interactive query loop
        print_separator('=')
        print("INTERACTIVE QUERY MODE")
        print_separator('=')
        print("\nEnter your questions below.")
        print("Commands:")
        print("  - Type 'exit' or 'quit' to exit")
        print("  - Type 'help' for help")
        print("  - Type 'stats' to see compression statistics")
        print()
        
        while True:
            try:
                # Get user query
                query = input("\nAsk question: ").strip()
                
                # Handle commands
                if query.lower() in ['exit', 'quit', 'q']:
                    print("\nExiting query engine. Goodbye!")
                    break
                
                if query.lower() == 'help':
                    print("\nQuery Engine Help:")
                    print("  - Enter any question to search compressed facts")
                    print("  - Results are ranked by semantic similarity")
                    print("  - After results, you can request source text")
                    print("  - Commands: 'exit', 'quit', 'help', 'stats'")
                    continue
                
                if query.lower() == 'stats':
                    stats = engine.get_compression_stats()
                    print("\nCompression Statistics:")
                    print(f"  Total Facts: {stats.get('total_facts', 0)}")
                    print(f"  Retained Facts: {stats.get('retained_facts', 0)}")
                    print(f"  Dropped Facts: {stats.get('dropped_facts', 0)}")
                    print(f"  Compression Ratio: {stats.get('compression_ratio', 0.0):.2%}")
                    print(f"  Information Retention: {stats.get('information_retention_score', 0.0):.2%}")
                    print(f"  Importance Loss: {stats.get('importance_loss', 0.0):.2%}")
                    continue
                
                if not query:
                    print("Please enter a question or command.")
                    continue
                
                # Execute query
                print(f"\nSearching for: '{query}'...")
                results = engine.query(query, top_k=5)
                
                if not results:
                    print("No matching facts found.")
                    continue
                
                # Display results
                print(f"\nFound {len(results)} matching fact(s):")
                print_separator()
                
                for i, fact in enumerate(results, 1):
                    display_fact(fact, index=i)
                
                print_separator()
                
                # Offer drill-down
                if drilldown_manager:
                    drilldown_choice = input("\nShow original source for top result? (yes/no): ").strip().lower()
                    
                    if drilldown_choice in ['yes', 'y']:
                        top_fact = results[0]
                        source_info = engine.get_source_text(top_fact)
                        
                        if source_info:
                            display_source(source_info)
                        else:
                            print("\nSource text not available for this fact.")
                            print("(Document structure may not be loaded)")
                
            except KeyboardInterrupt:
                print("\n\nExiting query engine. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
