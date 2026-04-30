from flask import Blueprint, request, jsonify

from app.services import document_store
from app.services.llm import chat

bp = Blueprint('query', __name__)

MAX_QUERY_PAPER_CHARS = 18_000


@bp.route('/api/query', methods=['POST'])
def query():
    body = request.get_json(silent=True) or {}
    doc_id = body.get('doc_id', '')
    question = body.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    doc = document_store.get(doc_id)
    if doc is None:
        return jsonify({'error': 'Document not found'}), 404

    full_text = doc.get('full_text', '') or ''
    truncated = len(full_text) > MAX_QUERY_PAPER_CHARS
    body_text = full_text[:MAX_QUERY_PAPER_CHARS]
    if truncated:
        body_text += '\n\n[... truncated: paper exceeds input-size limit for this request; earlier sections are included.]'

    system = """You are an academic reading assistant. The user message includes parsed paper text (may be truncated for length).
Answer only from that text. Be concise; cite sections when possible. Confidence 0.0–1.0.
Format:
ANSWER: <your answer>
CONFIDENCE: <0.0-1.0>"""

    meta = doc.get('metadata', {})
    title = meta.get('title', 'Unknown')
    user_msg = f"""Paper: {title}

Full paper text:
{body_text}

Question: {question}"""

    try:
        raw = chat(system, user_msg, max_tokens=2048)
    except Exception as e:
        return jsonify({'error': f'LLM call failed: {e}'}), 502

    answer, confidence = _parse_response(raw)
    return jsonify({'answer': answer, 'confidence': confidence})


def _parse_response(raw: str) -> tuple[str, float]:
    answer = raw
    confidence = 0.7
    if 'ANSWER:' in raw:
        parts = raw.split('CONFIDENCE:')
        answer = parts[0].replace('ANSWER:', '').strip()
        if len(parts) > 1:
            try:
                confidence = float(parts[1].strip()[:4])
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass
    return answer, confidence
