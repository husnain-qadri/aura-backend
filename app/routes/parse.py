from flask import Blueprint, request, jsonify

from app.services import document_store, paper_parser

bp = Blueprint('parse', __name__)


@bp.route('/api/parse', methods=['POST'])
def parse_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    pdf_bytes = file.read()
    if not pdf_bytes:
        return jsonify({'error': 'Empty file'}), 400

    doc_id = document_store.doc_id_from_bytes(pdf_bytes)

    cached = document_store.get(doc_id)
    if cached is not None:
        return jsonify({'doc_id': doc_id, **cached})

    try:
        parsed = paper_parser.parse_pdf(pdf_bytes)
    except Exception as e:
        return jsonify({'error': f'Parsing failed: {e}'}), 500

    document_store.put(doc_id, parsed)
    return jsonify({'doc_id': doc_id, **parsed})
