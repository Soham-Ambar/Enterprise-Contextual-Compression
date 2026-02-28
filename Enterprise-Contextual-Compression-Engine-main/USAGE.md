# Complete Usage Guide - Contextual Compression Engine

This guide provides step-by-step instructions for using the Enterprise Contextual Compression Engine.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Command-Line Usage](#command-line-usage)
4. [Programmatic Usage](#programmatic-usage)
5. [Understanding Output](#understanding-output)
6. [Common Use Cases](#common-use-cases)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: Check Python Version

```bash
python --version
# Should be Python 3.8 or higher
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Note**: This will install:
- PyTorch (for neural networks)
- sentence-transformers (for embeddings)
- pdfplumber/PyPDF2 (for PDF processing)
- Other dependencies

**First-time setup may take 5-10 minutes** as it downloads models and dependencies.

### Step 3: Verify Installation

```bash
# Test that imports work
python -c "from src.main import ContextualCompressionEngine; print('✓ Installation successful')"
```

---

## Quick Start

### Option 0: Web Interface (recommended)

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser. The landing page explains how to:

- Upload documents and tune compression settings
- Download compressed JSON
- Open the interactive query page to search facts and view source paragraphs

The web UI stores results in the `uploads/` folder and provides direct links between compression and query.

### Option 1: Run the Demo (Easiest)

```bash
# Run the pre-configured demo
python demo.py
```

This will:
- Load `sample.txt` (included enterprise policy document)
- Process it through the complete pipeline
- Display results in the terminal
- Save output to `compressed_output.json`

**Expected output:**
```
================================================================================
  ENTERPRISE CONTEXTUAL COMPRESSION ENGINE - DEMO
================================================================================

[Processing steps...]

Total Facts Extracted: 47
Important Facts Selected: 23
Compression Ratio: 48.94%
```

### Option 2: Process Your Own Document

```bash
# Process a single document
python src/main.py your_document.txt -o output.json

# Process a PDF
python src/main.py your_document.pdf -o output.json
```

---

## Command-Line Usage

### Basic Syntax

```bash
python src/main.py <input_file(s)> [options]
```

### Single Document Processing

```bash
# Basic usage (uses default threshold 0.5)
python src/main.py document.txt -o output.json

# With custom threshold (higher = more selective)
python src/main.py document.txt -o output.json -t 0.7

# With auto-tuning (automatically finds optimal threshold)
python src/main.py document.txt -o output.json --auto-tune --target-ratio 0.4

# With minimum confidence requirement
python src/main.py document.txt -o output.json --min-confidence 0.5
```

### Multiple Document Processing

```bash
# Process multiple documents and combine results
python src/main.py doc1.txt doc2.pdf doc3.txt -o combined_output.json

# With auto-tuning for optimal compression
python src/main.py policy1.pdf policy2.pdf --auto-tune --target-ratio 0.5
```

### All Command-Line Options

```bash
python src/main.py <input_files> [options]

Required:
  input_files              One or more document files (txt or pdf)

Optional:
  -o, --output PATH        Output JSON file path
                          (default: <input>_compressed.json)

  -t, --threshold FLOAT    Importance threshold (0.0-1.0)
                          Higher = more selective
                          (default: 0.5)

  --auto-tune              Automatically tune threshold for optimal compression

  --target-ratio FLOAT     Target compression ratio for auto-tuning (0.0-1.0)
                          (default: 0.5)

  --min-confidence FLOAT   Minimum confidence score (0.0-1.0)
                          (default: 0.4)

  --use-combined-score     Use combined score (LNN + extraction confidence)
                          (default: True)

  -m, --model PATH         Path to pre-trained LNN model weights
                          (optional)
```

### Examples

```bash
# Example 1: Process policy document with strict filtering
python src/main.py policy.pdf -o strict_output.json -t 0.8 --min-confidence 0.6

# Example 2: Process multiple documents with auto-tuning
python src/main.py doc1.txt doc2.txt doc3.txt -o combined.json --auto-tune

# Example 3: Process with custom output location
python src/main.py document.txt -o C:\outputs\compressed.json -t 0.6

# Example 4: Process PDF with high confidence requirement
python src/main.py report.pdf -o report_facts.json --min-confidence 0.7
```

---

## Programmatic Usage

### Basic Example

```python
from src.main import ContextualCompressionEngine

# Initialize the engine
engine = ContextualCompressionEngine(
    importance_threshold=0.5,
    use_combined_score=True,
    min_confidence=0.4
)

# Process a single document
result = engine.process_document(
    file_path='document.txt',
    output_path='output.json'
)

# Access results
print(f"Total facts: {result['metadata']['compression_stats']['total_facts']}")
print(f"Selected facts: {result['metadata']['compression_stats']['selected_facts']}")
```

### Advanced Example with Auto-Tuning

```python
from src.main import ContextualCompressionEngine

# Initialize with auto-tuning enabled
engine = ContextualCompressionEngine(
    importance_threshold=0.5,  # Initial threshold (will be tuned)
    auto_tune_threshold=True,
    target_compression_ratio=0.4,  # Keep top 40% of facts
    use_combined_score=True,
    min_confidence=0.4
)

# Process document
result = engine.process_document('policy.pdf', 'output.json')

# Access compressed facts
for fact in result['compressed_facts'][:5]:  # Top 5 facts
    print(f"Score: {fact['combined_score']:.3f}")
    print(f"Fact: {fact['fact']}")
    print(f"Type: {fact['fact_type']}")
    print(f"Source: {fact['source']['section_id']}")
    print()
```

### Multiple Document Processing

```python
from src.main import ContextualCompressionEngine

engine = ContextualCompressionEngine(importance_threshold=0.5)

# Process multiple documents
file_paths = ['policy1.pdf', 'policy2.txt', 'policy3.pdf']
result = engine.process_multiple_documents(
    file_paths=file_paths,
    output_path='combined_output.json'
)

# Access combined results
print(f"Total documents: {result['metadata']['total_documents']}")
print(f"Total facts: {result['metadata']['total_facts']}")

# Facts are sorted by importance across all documents
for fact in result['compressed_facts'][:10]:
    print(f"{fact['fact']} (Score: {fact['combined_score']:.3f})")
```

### Accessing Individual Components

```python
from src.main import ContextualCompressionEngine
from src.threshold_tuner import ThresholdTuner
from src.extraction import FactExtractor

# Initialize engine
engine = ContextualCompressionEngine()

# Access individual components
extractor = engine.extractor
tuner = engine.tuner

# Use threshold tuner directly
facts = [...]  # Your scored facts
optimal_threshold, stats = tuner.find_optimal_threshold(
    facts,
    target_compression_ratio=0.5
)
print(f"Optimal threshold: {optimal_threshold}")
```

---

## Understanding Output

### Output JSON Structure

```json
{
  "compressed_facts": [
    {
      "fact": "The maximum daily withdrawal limit for personal accounts is $5,000.",
      "importance_score": 0.9123,
      "confidence_score": 0.9500,
      "combined_score": 0.9136,
      "fact_type": "threshold",
      "source": {
        "document_id": "sample",
        "section_id": "sample_section_0",
        "paragraph_id": "sample_section_0_para_0"
      }
    }
  ],
  "metadata": {
    "document_id": "sample",
    "compression_stats": {
      "total_facts": 47,
      "selected_facts": 23,
      "compression_ratio": 0.489
    }
  }
}
```

### Field Explanations

- **`fact`**: The extracted fact text (verbatim from document)
- **`importance_score`**: LNN importance score (0.0-1.0)
- **`confidence_score`**: Extraction confidence (0.0-1.0)
- **`combined_score`**: Weighted combination (70% importance + 30% confidence)
- **`fact_type`**: Type of fact (threshold, risk, constraint, exception, compliance, number)
- **`source`**: Traceability information (document, section, paragraph IDs)

### Reading the Output

```python
import json

# Load output
with open('output.json', 'r') as f:
    result = json.load(f)

# Get statistics
stats = result['metadata']['compression_stats']
print(f"Extracted {stats['total_facts']} facts")
print(f"Selected {stats['selected_facts']} important facts")
print(f"Compression ratio: {stats['compression_ratio']:.1%}")

# Iterate through facts
for i, fact in enumerate(result['compressed_facts'], 1):
    print(f"\n{i}. {fact['fact']}")
    print(f"   Score: {fact['combined_score']:.3f}")
    print(f"   Type: {fact['fact_type']}")
    print(f"   Source: {fact['source']['section_id']}")
```

---

## Common Use Cases

### Use Case 1: Extract Compliance Rules from Policy Document

```bash
# Focus on compliance-related facts with high confidence
python src/main.py compliance_policy.pdf \
    -o compliance_rules.json \
    -t 0.6 \
    --min-confidence 0.7
```

### Use Case 2: Find All Financial Thresholds

```bash
# Lower threshold to capture more threshold facts
python src/main.py financial_policy.txt \
    -o thresholds.json \
    -t 0.4
```

### Use Case 3: Process Multiple Policy Documents

```bash
# Combine facts from multiple documents
python src/main.py policy1.pdf policy2.pdf policy3.pdf \
    -o all_policies.json \
    --auto-tune \
    --target-ratio 0.5
```

### Use Case 4: Extract Risk Clauses

```python
from src.main import ContextualCompressionEngine

engine = ContextualCompressionEngine(importance_threshold=0.5)
result = engine.process_document('risk_assessment.pdf', 'risks.json')

# Filter for risk facts only
risk_facts = [
    f for f in result['compressed_facts']
    if f['fact_type'] == 'risk'
]

print(f"Found {len(risk_facts)} risk-related facts")
```

### Use Case 5: Batch Processing with Custom Thresholds

```python
from src.main import ContextualCompressionEngine
import os

# Process all PDFs in a directory
pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]

engine = ContextualCompressionEngine(importance_threshold=0.6)
result = engine.process_multiple_documents(
    file_paths=pdf_files,
    output_path='batch_output.json'
)
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Make sure you're in the project directory
cd "C:\Users\vivek\OneDrive\Desktop\cursor projects\PS4"

# Install dependencies
pip install -r requirements.txt
```

### Issue: "File not found"

**Solution:**
- Use absolute paths or ensure you're in the correct directory
- Check file extensions (.txt or .pdf)
- Verify file permissions

```bash
# Use absolute path
python src/main.py "C:\path\to\document.pdf" -o output.json
```

### Issue: "CUDA out of memory" (if using GPU)

**Solution:**
- The system automatically falls back to CPU
- Or set device manually in code:
```python
import torch
torch.cuda.set_device(0)  # Use GPU 0
```

### Issue: PDF extraction fails

**Solution:**
- The system has fallback mechanisms (pdfplumber → PyPDF2)
- If both fail, try converting PDF to text first
- Check if PDF is password-protected or corrupted

### Issue: Too many/few facts selected

**Solution:**
- Adjust threshold: lower threshold = more facts, higher = fewer
- Use auto-tuning: `--auto-tune --target-ratio 0.5`
- Adjust confidence: `--min-confidence 0.3` (lower = more facts)

```bash
# Get more facts
python src/main.py document.txt -o output.json -t 0.3

# Get fewer, higher-quality facts
python src/main.py document.txt -o output.json -t 0.7 --min-confidence 0.6
```

### Issue: Slow processing

**Solution:**
- First run downloads models (~500MB) - subsequent runs are faster
- Large documents take longer - this is normal
- Processing speed: ~100-200 facts/second on CPU

---

## Quick Reference

### Most Common Commands

```bash
# Quick demo
python demo.py

# Single document
python src/main.py document.txt -o output.json

# Multiple documents
python src/main.py doc1.txt doc2.pdf -o combined.json

# With auto-tuning
python src/main.py document.txt -o output.json --auto-tune

# High-quality facts only
python src/main.py document.txt -o output.json -t 0.7 --min-confidence 0.6
```

### Python Quick Reference

```python
from src.main import ContextualCompressionEngine

# Initialize
engine = ContextualCompressionEngine(importance_threshold=0.5)

# Single document
result = engine.process_document('doc.txt', 'output.json')

# Multiple documents
result = engine.process_multiple_documents(['doc1.txt', 'doc2.pdf'], 'output.json')

# Access facts
facts = result['compressed_facts']
stats = result['metadata']['compression_stats']
```

---

## Next Steps

1. **Run the demo**: `python demo.py`
2. **Process your document**: `python src/main.py your_doc.txt -o output.json`
3. **Explore output**: Open `output.json` to see compressed facts
4. **Experiment with thresholds**: Try different `-t` values
5. **Try auto-tuning**: Use `--auto-tune` for optimal compression

For more details, see the [README.md](README.md) file.
