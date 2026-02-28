from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import logging

# make sure src is importable
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main import ContextualCompressionEngine
from query_engine import QueryEngine
from drilldown import DrillDownManager
from traceability import TraceabilityManager
from ingestion import DocumentIngester
from chunking import HierarchicalChunker

# basic logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# directory where uploaded files and outputs will be saved
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'doc' not in request.files:
            return "No file part", 400
        file = request.files['doc']
        if file.filename == '':
            return "No selected file", 400
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        # read UI parameters
        try:
            threshold = float(request.form.get('threshold', 0.5))
        except Exception:
            threshold = 0.5
        try:
            min_confidence = float(request.form.get('min_confidence', 0.4))
        except Exception:
            min_confidence = 0.4
        use_combined = request.form.get('use_combined', 'on') in ('on', 'true', 'True', '1')
        model_path = request.form.get('model_path') or None
        auto_tune = request.form.get('auto_tune', 'off') in ('on', 'true', 'True', '1')
        try:
            target_ratio = float(request.form.get('target_ratio', 0.5))
        except Exception:
            target_ratio = 0.5

        # prepare output path for compressed JSON
        output_filename = f"{Path(filename).stem}_compressed.json"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        # run compression with a fresh engine configured from the UI
        try:
            engine = ContextualCompressionEngine(
                importance_threshold=threshold,
                model_path=model_path,
                use_combined_score=use_combined,
                min_confidence=min_confidence,
                auto_tune_threshold=auto_tune,
                target_compression_ratio=target_ratio
            )

            result = engine.process_document(path, output_path=output_path)
        except Exception as e:
            logging.exception("Compression failed")
            return f"Error processing document: {e}", 500

        # remove the uploaded raw file to keep uploads folder tidy (keep output JSON)
        try:
            os.remove(path)
        except Exception:
            pass

        return render_template('results.html', result=result, result_file=output_filename)

    return render_template('upload.html')


@app.route('/')
def landing():
    """Landing page shown when the app starts."""
    return render_template('landing.html')


@app.route('/download/<path:filename>')
def download(filename):
    # Serve compressed JSON from the uploads folder as an attachment
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


def list_compressed_files():
    files = []
    for fname in os.listdir(app.config['UPLOAD_FOLDER']):
        if fname.endswith('.json'):
            files.append(fname)
    files.sort(key=lambda n: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], n)), reverse=True)
    return files


@app.route('/query', methods=['GET', 'POST'])
def query_ui():
    files = list_compressed_files()
    selected = request.args.get('file') or (files[0] if files else None)

    results = None
    metadata = None
    stats = None

    if request.method == 'POST':
        selected = request.form.get('file')
        question = request.form.get('question', '').strip()
        top_k = int(request.form.get('top_k', 5))

        if not selected:
            return "No compressed result file available", 400

        compressed_path = os.path.join(app.config['UPLOAD_FOLDER'], selected)

        try:
            # Initialize query engine with the selected compressed JSON
            qengine = QueryEngine(compressed_data_path=compressed_path, top_k=top_k)

            # Try to enable drill-down by loading original document structure
            metadata = qengine.get_metadata()
            source_file = metadata.get('source_file') or None
            # Try to find a matching original file by document_id if no explicit source
            if not source_file:
                doc_id = metadata.get('document_id')
                if doc_id:
                    possible = [f"{doc_id}.txt", f"{doc_id}.pdf", os.path.join('uploads', f"{doc_id}.txt")]
                    for p in possible:
                        if os.path.exists(p):
                            source_file = p
                            break

            if source_file and os.path.exists(source_file):
                ingester = DocumentIngester()
                chunker = HierarchicalChunker()
                traceability = TraceabilityManager()

                document = ingester.load_document(source_file)
                structured = chunker.chunk_document(document)

                dd = DrillDownManager(traceability)
                dd.register_document_structure(structured['document_id'], structured)
                # attach traceability for fallback
                dd.traceability = traceability
                qengine.set_drilldown_manager(dd)

            # Run the query
            results = qengine.query(question, top_k=top_k)

            # For each result, attempt to get source text
            for r in results:
                src = qengine.get_source_text(r)
                r['source_text'] = src.get('paragraph_text') if src else None
                r['section_title'] = src.get('section_title') if src else None

            stats = qengine.get_compression_stats()

        except Exception as e:
            logging.exception("Query failed")
            return f"Error running query: {e}", 500

    return render_template('query.html', files=files, selected=selected, results=results, metadata=metadata, stats=stats)


if __name__ == '__main__':
    app.run(debug=True)
